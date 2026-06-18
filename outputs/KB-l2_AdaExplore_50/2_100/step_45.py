import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_gather_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    MIN_VAL, INV_DIV,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
    IC_BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    SP = OD * OH * OW
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < SP

    od = sp_offs // (OH * OW)
    rem = sp_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # iterate over kernel
    for kd in tl.static_range(KD):
        id_pos = od + PAD - kd
        id_valid = (id_pos % STRIDE) == 0
        id_idx = id_pos // STRIDE
        id_in_range = (id_idx >= 0) & (id_idx < ID) & id_valid
        for kh in tl.static_range(KH):
            ih_pos = oh + PAD - kh
            ih_valid = (ih_pos % STRIDE) == 0
            ih_idx = ih_pos // STRIDE
            ih_in_range = (ih_idx >= 0) & (ih_idx < IH) & ih_valid
            for kw in tl.static_range(KW):
                iw_pos = ow + PAD - kw
                iw_valid = (iw_pos % STRIDE) == 0
                iw_idx = iw_pos // STRIDE
                iw_in_range = (iw_idx >= 0) & (iw_idx < IW) & iw_valid

                spatial_valid = id_in_range & ih_in_range & iw_in_range & sp_mask

                # base input offset for this (kd,kh,kw) per spatial position
                # x layout: (N, IC, ID, IH, IW) contiguous
                in_spatial_off = id_idx * (IH * IW) + ih_idx * IW + iw_idx  # [BLOCK_SP]

                # weight layout pre-transposed to (KD, KH, KW, IC, OC)
                w_base = ((kd * KH + kh) * KW + kw) * IC * OC

                # loop over IC
                for ic_start in range(0, IC, IC_BLOCK):
                    ic_offs = ic_start + tl.arange(0, IC_BLOCK)
                    ic_mask = ic_offs < IC

                    # load x: shape [IC_BLOCK, BLOCK_SP]
                    x_off = (pid_n * IC + ic_offs[:, None]) * (ID * IH * IW) + in_spatial_off[None, :]
                    x_mask = ic_mask[:, None] & spatial_valid[None, :]
                    x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                    # load w: shape [IC_BLOCK, BLOCK_OC]
                    w_off = w_base + ic_offs[:, None] * OC + oc_offs[None, :]
                    w_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                    # acc += w.T @ x  -> [BLOCK_OC, BLOCK_SP]
                    acc += tl.dot(tl.trans(w_vals), x_vals)

    # add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    # clamp + divide
    acc = tl.where(acc < MIN_VAL, MIN_VAL, acc)
    acc = acc * INV_DIV

    # store: out layout (N, OC, OD, OH, OW)
    out_off = (pid_n * OC + oc_offs[:, None]) * SP + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.min_value = float(min_value)
        self.divisor = float(divisor)

        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Original weight shape: (IC, OC, KD, KH, KW)
        w = conv.weight.detach().clone()
        b = conv.bias.detach().clone()

        # Pre-transpose weight to (KD, KH, KW, IC, OC) layout
        w_perm = w.permute(2, 3, 4, 0, 1).contiguous()
        self.register_buffer('weight_t', w_perm)
        self.register_buffer('bias', b)

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        STRIDE = self.stride
        PAD = self.padding
        OC = self.out_channels

        OD = (ID - 1) * STRIDE - 2 * PAD + KD
        OH = (IH - 1) * STRIDE - 2 * PAD + KH
        OW = (IW - 1) * STRIDE - 2 * PAD + KW

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_SP = 64
        IC_BLOCK = 32

        SP = OD * OH * OW
        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(SP, BLOCK_SP))

        conv_transpose3d_gather_kernel[grid](
            x, self.weight_t, self.bias, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            STRIDE, PAD,
            self.min_value, 1.0 / self.divisor,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            IC_BLOCK=IC_BLOCK,
            num_warps=4, num_stages=2,
        )
        return out