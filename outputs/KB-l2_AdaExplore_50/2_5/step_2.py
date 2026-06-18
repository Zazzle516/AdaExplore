import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ConvTranspose2d as gather form:
# For each output pixel (oh, ow), and each (kh, kw) tap where
#   (oh + pad - kh) % stride == 0 and 0 <= ih < IH and 0 <= iw < IW
# we accumulate sum_{ic} input[n, ic, ih, iw] * weight[ic, oc, kh, kw]
#
# We tile over (N, OC_block, output_spatial_block).
# Treat reduction over ic as the inner GEMM-K dimension.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
    ],
    key=['IC', 'OC', 'IH', 'IW', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv_transpose2d_gather_kernel(
    inp_ptr,    # [N, IC, IH, IW]
    weight_ptr, # [IC, OC, KH, KW]
    bias_ptr,   # [OC]
    out_ptr,    # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,  # output spatial tile
    BLOCK_N: tl.constexpr,  # OC tile
    BLOCK_K: tl.constexpr,  # IC tile
):
    pid_n = tl.program_id(0)  # batch
    pid_m = tl.program_id(1)  # output spatial block
    pid_oc = tl.program_id(2) # oc block

    # output spatial indices for this block
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    oh = m_offs // OW
    ow = m_offs % OW
    m_mask = m_offs < (OH * OW)

    oc_offs = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over kernel taps
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # ih_num = oh + PAD - kh; valid if ih_num % STRIDE == 0 and 0 <= ih_num/STRIDE < IH
            ih_num = oh + PAD - kh
            iw_num = ow + PAD - kw
            ih = ih_num // STRIDE
            iw = iw_num // STRIDE
            valid_h = (ih_num >= 0) & ((ih_num - ih * STRIDE) == 0) & (ih < IH)
            valid_w = (iw_num >= 0) & ((iw_num - iw * STRIDE) == 0) & (iw < IW)
            valid = valid_h & valid_w & m_mask  # [BLOCK_M]

            # reduce over ic in blocks
            for k0 in range(0, IC, BLOCK_K):
                k_offs = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
                k_mask = k_offs < IC

                # Load input[n, k, ih, iw] -> shape [BLOCK_M, BLOCK_K]
                # input stride: n*IC*IH*IW + k*IH*IW + ih*IW + iw
                inp_base = pid_n * IC * IH * IW
                inp_addr = (inp_base
                            + k_offs[None, :] * (IH * IW)
                            + ih[:, None] * IW
                            + iw[:, None])
                inp_load_mask = valid[:, None] & k_mask[None, :]
                a = tl.load(inp_ptr + inp_addr, mask=inp_load_mask, other=0.0)

                # Load weight[k, oc, kh, kw] -> shape [BLOCK_K, BLOCK_N]
                # weight stride: k*OC*KH*KW + oc*KH*KW + kh*KW + kw
                w_addr = (k_offs[:, None] * (OC * KH * KW)
                          + oc_offs[None, :] * (KH * KW)
                          + kh * KW + kw)
                w_load_mask = k_mask[:, None] & oc_mask[None, :]
                w = tl.load(weight_ptr + w_addr, mask=w_load_mask, other=0.0)

                acc += tl.dot(a, w)

    # Epilogue: subtract bias (shape OC), apply tanh
    b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_N]
    acc = acc - b[None, :]
    # tanh
    acc = tl.where(acc > 20.0, 1.0, tl.where(acc < -20.0, -1.0,
              (tl.exp(2.0 * acc) - 1.0) / (tl.exp(2.0 * acc) + 1.0)))

    # store output[n, oc, oh, ow]
    out_base = pid_n * OC * OH * OW
    out_addr = (out_base
                + oc_offs[None, :] * (OH * OW)
                + oh[:, None] * OW
                + ow[:, None])
    out_mask = m_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_addr, acc, mask=out_mask)


def conv_transpose2d_fused(x, weight, conv_bias, extra_bias, stride, padding, output_padding):
    """
    x:      [N, IC, IH, IW]
    weight: [IC, OC, KH, KW]
    conv_bias: [OC] or None  (the nn.ConvTranspose2d's built-in bias)
    extra_bias: [OC] (the subtracted bias)
    Returns tanh(convT(x) + conv_bias - extra_bias).
    """
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    # effective bias to subtract: extra_bias - conv_bias
    if conv_bias is not None:
        eff_bias = (extra_bias - conv_bias).contiguous()
    else:
        eff_bias = extra_bias.contiguous()

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        N,
        triton.cdiv(OH * OW, META['BLOCK_M']),
        triton.cdiv(OC, META['BLOCK_N']),
    )

    conv_transpose2d_gather_kernel[grid](
        x, weight, eff_bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, padding,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        cb = self.conv_transpose.bias
        eb = self.bias.view(-1).contiguous()
        return conv_transpose2d_fused(
            x, w, cb, eb,
            self.stride, self.padding, self.output_padding,
        )