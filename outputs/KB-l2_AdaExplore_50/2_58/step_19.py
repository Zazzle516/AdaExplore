import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_lse_hs_bias_clamp_kernel(
    x_ptr,         # (N, IC, ID, IH, IW)
    w_ptr,         # (IC, OC, KD, KH, KW)
    cb_ptr,        # (OC,) conv bias
    bias_ptr,      # 0-d
    out_ptr,       # (N, 1, OD, OH, OW)
    N, IC: tl.constexpr, OC: tl.constexpr,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    total_sp = OD * OH * OW
    sp_mask = sp_offs < total_sp

    od = sp_offs // (OH * OW)
    rem = sp_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    bias_val = tl.load(bias_ptr)

    # output buffer per channel? Instead, compute conv outputs on the fly via accumulator [BLOCK_SP, OC].
    acc = tl.zeros([BLOCK_SP, OC], dtype=tl.float32)

    # Add conv bias broadcast
    oc_range = tl.arange(0, OC)
    cb = tl.load(cb_ptr + oc_range)  # (OC,)
    acc += cb[None, :]

    # For each (kd, kh, kw), find input position id, ih, iw such that:
    # od = id*STRIDE - PAD + kd  =>  id = (od + PAD - kd) / STRIDE
    # similarly for ih, iw. Must be integer and in range.
    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_q = id_num // STRIDE
        id_valid = (id_num % STRIDE == 0) & (id_q >= 0) & (id_q < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_q = ih_num // STRIDE
            ih_valid = (ih_num % STRIDE == 0) & (ih_q >= 0) & (ih_q < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PAD - kw
                iw_q = iw_num // STRIDE
                iw_valid = (iw_num % STRIDE == 0) & (iw_q >= 0) & (iw_q < IW)

                valid = id_valid & ih_valid & iw_valid & sp_mask  # (BLOCK_SP,)

                # input offset for each output position: n*IC*ID*IH*IW + ic*ID*IH*IW + id*IH*IW + ih*IW + iw
                in_base = pid_n * (IC * ID * IH * IW) + id_q * (IH * IW) + ih_q * IW + iw_q  # (BLOCK_SP,)

                # Loop over IC, accumulate weight outer product
                for ic in tl.static_range(0, IC):
                    in_off = in_base + ic * (ID * IH * IW)
                    x_val = tl.load(x_ptr + in_off, mask=valid, other=0.0)  # (BLOCK_SP,)

                    # weight: (IC, OC, KD, KH, KW): w_ptr + ic*OC*KD*KH*KW + oc*KD*KH*KW + kd*KH*KW + kh*KW + kw
                    w_off = ic * (OC * KD * KH * KW) + oc_range * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)  # (OC,)

                    acc += x_val[:, None] * w_val[None, :]

    # LSE over OC dim
    max_val = tl.max(acc, axis=1)  # (BLOCK_SP,)
    shifted = acc - max_val[:, None]
    sum_exp = tl.sum(tl.exp(shifted), axis=1)
    lse = max_val + tl.log(sum_exp)

    # HardSwish
    hs = lse * tl.sigmoid(lse + 3.0) / 6.0
    res = hs - bias_val
    res = tl.minimum(tl.maximum(res, -1.0), 1.0)

    out_off = pid_n * (OD * OH * OW) + sp_offs
    tl.store(out_ptr + out_off, res, mask=sp_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        STRIDE = self.stride
        PAD = self.padding

        OD = (ID - 1) * STRIDE - 2 * PAD + KD
        OH = (IH - 1) * STRIDE - 2 * PAD + KH
        OW = (IW - 1) * STRIDE - 2 * PAD + KW

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        cbias = self.conv_transpose.bias.contiguous()
        bias_flat = self.bias.view(-1)[0:1]

        out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

        total_sp = OD * OH * OW
        BLOCK_SP = 64
        grid = (N, (total_sp + BLOCK_SP - 1) // BLOCK_SP)

        conv_transpose3d_lse_hs_bias_clamp_kernel[grid](
            x, weight, cbias, bias_flat, out,
            N, IC, OC, ID, IH, IW, OD, OH, OW,
            KD, KH, KW, STRIDE, PAD,
            BLOCK_SP=BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )
        return out