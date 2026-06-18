import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, scale_ptr, bias2_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KT: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # program ids: (n*OD, OH_tile*OW_tile, OC_tile)
    pid_n_od = tl.program_id(0)
    pid_sp = tl.program_id(1)
    pid_oc = tl.program_id(2)

    n = pid_n_od // OD
    od = pid_n_od % OD

    # spatial tile across OH*OW
    sp_start = pid_sp * BLOCK_M
    sp_offs = sp_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    oh = sp_offs // OW
    ow = sp_offs % OW
    sp_mask = sp_offs < (OH * OW)

    # OC tile
    oc_offs = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kernel positions and IC
    # weight layout: (OC, IC, KT, KH, KW)
    # input layout: (N, IC, ID, IH, IW)
    for kt in tl.static_range(0, KT):
        id_ = od + kt
        for kh in tl.static_range(0, KH):
            ih = oh + kh  # [BLOCK_M]
            for kw in tl.static_range(0, KW):
                iw = ow + kw  # [BLOCK_M]
                # load IC vector for each spatial point: shape [BLOCK_M, IC_C]
                ic_range = tl.arange(0, IC_C)  # [IC_C]
                # input addr: n*IC*ID*IH*IW + ic*ID*IH*IW + id_*IH*IW + ih*IW + iw
                in_base = n * IC * ID * IH * IW + id_ * IH * IW
                in_addr = (in_base
                           + ic_range[None, :] * (ID * IH * IW)
                           + ih[:, None] * IW
                           + iw[:, None])  # [BLOCK_M, IC_C]
                in_mask = sp_mask[:, None] & (ic_range[None, :] < IC)
                x_vals = tl.load(x_ptr + in_addr, mask=in_mask, other=0.0)  # [BLOCK_M, IC_C]

                # weight addr: oc*IC*KT*KH*KW + ic*KT*KH*KW + kt*KH*KW + kh*KW + kw
                w_addr = (oc_offs[None, :] * (IC * KT * KH * KW)
                          + ic_range[:, None] * (KT * KH * KW)
                          + kt * KH * KW + kh * KW + kw)  # [IC_C, BLOCK_N]
                w_mask = (ic_range[:, None] < IC) & oc_mask[None, :]
                w_vals = tl.load(w_ptr + w_addr, mask=w_mask, other=0.0)  # [IC_C, BLOCK_N]

                acc += tl.dot(x_vals, w_vals)

    # bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_N]
    acc = acc + b_vals[None, :]

    # scale (per OC)
    s_vals = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    y = acc * s_vals[None, :]
    # tanh
    e2 = tl.exp(2.0 * y)
    t = (e2 - 1.0) / (e2 + 1.0)
    # bias2 mul
    b2 = tl.load(bias2_ptr + oc_offs, mask=oc_mask, other=0.0)
    z = t * b2[None, :]
    # sigmoid
    out = 1.0 / (1.0 + tl.exp(-z))

    # store: out shape (N, OC, OD, OH, OW)
    out_addr = (n * OC * OD * OH * OW
                + oc_offs[None, :] * (OD * OH * OW)
                + od * (OH * OW)
                + sp_offs[:, None])
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_addr, out, mask=out_mask)


def fused_conv3d(x, weight, bias, scale, bias2):
    N, IC, ID, IH, IW = x.shape
    OC, _, KT, KH, KW = weight.shape
    OD = ID - KT + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    scale_flat = scale.contiguous().view(-1)
    bias2_flat = bias2.contiguous().view(-1)

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 16
    IC_C = 4 if IC <= 4 else 8  # power of 2 covering IC

    grid = (N * OD, (OH * OW + BLOCK_M - 1) // BLOCK_M, (OC + BLOCK_N - 1) // BLOCK_N)

    conv3d_fused_kernel[grid](
        x, weight, bias, scale_flat, bias2_flat, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KT=KT, KH=KH, KW=KW,
        IC_C=IC_C,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return fused_conv3d(
            x,
            self.conv.weight,
            self.conv.bias,
            self.scaling_factor,
            self.bias,
        )