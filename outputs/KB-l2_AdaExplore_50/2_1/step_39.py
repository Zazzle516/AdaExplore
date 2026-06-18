import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'M_TOTAL', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_nhwc_relu_bias_kernel(
    x_ptr,         # NHWC input: [N, IH, IW, IC]
    w_ptr,         # weight reshaped: [OC, KH, KW, IC]
    b_ptr,         # conv bias [OC]
    bias_ptr,      # extra bias [OC]
    out_ptr,       # NCHW output: [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    M_TOTAL,       # N * OH * OW
    K_TOTAL,       # KH * KW * IC
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile over M = N*OH*OW
    pid_n = tl.program_id(1)  # tile over N = OC

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decompose m into (n, oh, ow)
    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    m_mask = offs_m < M_TOTAL
    n_mask = offs_n < OC

    # Base x pointer per row
    x_base = n_idx * (IH * IW * IC)  # [BLOCK_M]

    offs_k = tl.arange(0, BLOCK_K)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # Per (kh,kw), input column base = (oh+kh)*IW*IC + (ow+kw)*IC
            ih = oh + kh
            iw = ow + kw
            x_row_base = x_base + ih * (IW * IC) + iw * IC  # [BLOCK_M]

            # Weight column base for this (kh,kw): w[oc, kh, kw, :]
            w_kk_base = (kh * KW + kw) * IC
            w_row_base = offs_n * K_TOTAL + w_kk_base  # [BLOCK_N]

            for ic_start in range(0, IC, BLOCK_K):
                ic_idx = ic_start + offs_k
                ic_mask = ic_idx < IC

                x_offs = x_row_base[:, None] + ic_idx[None, :]
                x_vals = tl.load(x_ptr + x_offs,
                                 mask=(m_mask[:, None]) & (ic_mask[None, :]),
                                 other=0.0)

                w_offs = w_row_base[:, None] + ic_idx[None, :]
                w_vals = tl.load(w_ptr + w_offs,
                                 mask=(n_mask[:, None]) & (ic_mask[None, :]),
                                 other=0.0)

                acc += tl.dot(x_vals, tl.trans(w_vals))

    # bias + relu + extra bias
    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    eb_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)

    acc = acc + b_vals[None, :]
    acc = tl.maximum(acc, 0.0)
    acc = acc + eb_vals[None, :]

    # Store NHWC output: out[n, oh, ow, oc] -- coalesced on channel axis
    out_offs = (n_idx[:, None] * (OH * OW * OC) +
                oh[:, None] * (OW * OC) +
                ow[:, None] * OC +
                offs_n[None, :])
    out_mask = (m_mask[:, None]) & (n_mask[None, :])
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv2d_relu_bias_nhwc(x_nchw, weight, conv_bias, extra_bias):
    N, IC, IH, IW = x_nchw.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    # NCHW -> NHWC
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

    # weight [OC, IC, KH, KW] -> [OC, KH, KW, IC] -> [OC, KH*KW*IC]
    w_reshaped = weight.permute(0, 2, 3, 1).contiguous().view(OC, KH * KW * IC)

    conv_bias_c = conv_bias.contiguous()
    extra_bias_flat = extra_bias.contiguous().view(-1)

    # NHWC output buffer
    out_nhwc = torch.empty((N, OH, OW, OC), device=x_nchw.device, dtype=x_nchw.dtype)

    M_TOTAL = N * OH * OW
    K_TOTAL = KH * KW * IC

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    conv2d_nhwc_relu_bias_kernel[grid](
        x_nhwc, w_reshaped, conv_bias_c, extra_bias_flat, out_nhwc,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        M_TOTAL, K_TOTAL,
    )

    # NHWC -> NCHW
    return out_nhwc.permute(0, 3, 1, 2).contiguous()


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.cuda()
        return conv2d_relu_bias_nhwc(x, self.conv.weight, self.conv.bias, self.bias)