import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


@triton.jit
def conv_transpose3d_gather_kernel(
    x_ptr,        # input: (N, ID, IH, IW, IC) channels_last
    w_ptr,        # weight: (KD, KH, KW, IC, OC)
    b_ptr,        # bias: (OC,)
    out_ptr,      # output: (N, OD, OH, OW, OC) channels_last
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    MIN_VAL, INV_DIV,
    BLOCK_M: tl.constexpr,  # spatial tile (output positions)
    BLOCK_N: tl.constexpr,  # OC tile
    IC_BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_oc = tl.program_id(1)  # OC tile
    pid_sp = tl.program_id(2)  # spatial tile

    sp_size = OD * OH * OW
    sp_off = pid_sp * BLOCK_M + tl.arange(0, BLOCK_M)
    sp_mask = sp_off < sp_size

    # decompose sp_off into od, oh, ow
    od = sp_off // (OH * OW)
    rem = sp_off % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_off = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)
    oc_mask = oc_off < OC

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate kernel positions
    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_valid_stride = (id_num % STRIDE) == 0
        id_idx = id_num // STRIDE
        id_in_bounds = (id_idx >= 0) & (id_idx < ID) & id_valid_stride

        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_valid_stride = (ih_num % STRIDE) == 0
            ih_idx = ih_num // STRIDE
            ih_in_bounds = (ih_idx >= 0) & (ih_idx < IH) & ih_valid_stride

            for kw in tl.static_range(0, KW):
                iw_num = ow + PAD - kw
                iw_valid_stride = (iw_num % STRIDE) == 0
                iw_idx = iw_num // STRIDE
                iw_in_bounds = (iw_idx >= 0) & (iw_idx < IW) & iw_valid_stride

                spatial_valid = id_in_bounds & ih_in_bounds & iw_in_bounds & sp_mask  # [BLOCK_M]

                # base input offset for each spatial position
                # x layout: (N, ID, IH, IW, IC), n stride = ID*IH*IW*IC
                in_base = (pid_n * ID * IH * IW + id_idx * IH * IW + ih_idx * IW + iw_idx) * IC  # [BLOCK_M]

                # weight layout: (KD, KH, KW, IC, OC), offset for (kd,kh,kw) is fixed
                w_base = ((kd * KH + kh) * KW + kw) * IC * OC  # scalar

                # GEMM-K loop over IC
                for ic_start in range(0, IC, IC_BLOCK):
                    ic_off = ic_start + tl.arange(0, IC_BLOCK)
                    ic_mask = ic_off < IC

                    # x: [BLOCK_M, IC_BLOCK]
                    x_ptrs = x_ptr + in_base[:, None] + ic_off[None, :]
                    x_vals = tl.load(
                        x_ptrs,
                        mask=spatial_valid[:, None] & ic_mask[None, :],
                        other=0.0,
                    )

                    # w: [IC_BLOCK, BLOCK_N]
                    w_ptrs = w_ptr + w_base + ic_off[:, None] * OC + oc_off[None, :]
                    w_vals = tl.load(
                        w_ptrs,
                        mask=ic_mask[:, None] & oc_mask[None, :],
                        other=0.0,
                    )

                    acc += tl.dot(x_vals, w_vals)

    # add bias
    bias = tl.load(b_ptr + oc_off, mask=oc_mask, other=0.0)
    acc = acc + bias[None, :]

    # clamp + div
    acc = tl.where(acc < MIN_VAL, MIN_VAL, acc)
    acc = acc * INV_DIV

    # store: output layout (N, OD, OH, OW, OC)
    out_base = (pid_n * OD * OH * OW + sp_off) * OC  # [BLOCK_M]
    out_ptrs = out_ptr + out_base[:, None] + oc_off[None, :]
    tl.store(out_ptrs, acc, mask=sp_mask[:, None] & oc_mask[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # weight shape: (IC, OC, KD, KH, KW)
        w = conv.weight.detach().clone()
        b = conv.bias.detach().clone() if conv.bias is not None else torch.zeros(out_channels)

        # transpose to (KD, KH, KW, IC, OC)
        w_t = w.permute(2, 3, 4, 0, 1).contiguous()

        self.register_buffer("weight_t", w_t.cuda())
        self.register_buffer("bias", b.cuda())

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.min_value = float(min_value)
        self.divisor = float(divisor)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        OD = (ID - 1) * S - 2 * P + KD
        OH = (IH - 1) * S - 2 * P + KH
        OW = (IW - 1) * S - 2 * P + KW
        OC = self.out_channels

        # convert input to channels_last_3d: (N, ID, IH, IW, IC)
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()

        # output in channels_last: (N, OD, OH, OW, OC)
        out_cl = torch.empty((N, OD, OH, OW, OC), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_N = 64
        IC_BLOCK = 32

        grid = (N, triton.cdiv(OC, BLOCK_N), triton.cdiv(OD * OH * OW, BLOCK_M))

        conv_transpose3d_gather_kernel[grid](
            x_cl, self.weight_t, self.bias, out_cl,
            N, IC, OC,
            ID, IH, IW,
            OD, OH, OW,
            KD, KH, KW,
            S, P,
            self.min_value, 1.0 / self.divisor,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, IC_BLOCK=IC_BLOCK,
            num_warps=4, num_stages=2,
        )

        # convert back to (N, OC, OD, OH, OW)
        out = out_cl.permute(0, 4, 1, 2, 3).contiguous()
        return out