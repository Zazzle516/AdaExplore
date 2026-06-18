import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'M_TOTAL', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_nhwc_relu_bias_kernel(
    x_ptr,         # NHWC input: [N, IH, IW, IC]
    w_ptr,         # weight reshaped: [KH*KW*IC, OC] (transposed for K-contig on N axis)
    b_ptr,         # conv bias [OC]
    bias_ptr,      # extra bias [OC]
    out_ptr,       # NCHW output: [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    M_TOTAL,       # N * OH * OW
    K_TOTAL,       # KH * KW * IC
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decompose m into (n, oh, ow)
    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    m_mask = offs_m < M_TOTAL
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Base input pointer offset for each m (independent of k)
    x_base = n_idx * (IH * IW * IC) + oh * (IW * IC) + ow * IC  # [BLOCK_M]

    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_TOTAL

        # K decompose as (kh, kw, ic) with ic innermost
        kh = offs_k // (KW * IC)
        kw = (offs_k // IC) % KW
        ic = offs_k % IC

        # Shift offset for kh/kw
        k_shift = kh * (IW * IC) + kw * IC + ic  # [BLOCK_K]

        x_offs = x_base[:, None] + k_shift[None, :]
        x_vals = tl.load(x_ptr + x_offs,
                         mask=(m_mask[:, None]) & (k_mask[None, :]),
                         other=0.0)

        # Weight is [K_TOTAL, OC] (transposed) so it's K-major, OC-contig
        w_offs = offs_k[:, None] * OC + offs_n[None, :]
        w_vals = tl.load(w_ptr + w_offs,
                         mask=(k_mask[:, None]) & (n_mask[None, :]),
                         other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # bias + relu + extra bias
    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    eb_vals = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)

    acc = acc + b_vals[None, :]
    acc = tl.maximum(acc, 0.0)
    acc = acc + eb_vals[None, :]

    # Store NCHW directly: out[n, oc, oh, ow]
    out_offs = (n_idx[:, None] * (OC * OH * OW) +
                offs_n[None, :] * (OH * OW) +
                oh[:, None] * OW +
                ow[:, None])
    out_mask = (m_mask[:, None]) & (n_mask[None, :])
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv2d_relu_bias_fused(x_nchw, weight, conv_bias, extra_bias,
                            x_nhwc_cache, w_cache):
    N, IC, IH, IW = x_nchw.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    # NCHW -> NHWC
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

    conv_bias_c = conv_bias.contiguous()
    extra_bias_flat = extra_bias.contiguous().view(-1)

    out = torch.empty((N, OC, OH, OW), device=x_nchw.device, dtype=x_nchw.dtype)

    M_TOTAL = N * OH * OW
    K_TOTAL = KH * KW * IC

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    conv2d_nhwc_relu_bias_kernel[grid](
        x_nhwc, w_cache, conv_bias_c, extra_bias_flat, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        M_TOTAL, K_TOTAL,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._w_cache = None
        self._w_version = None

    def _get_weight_cache(self):
        w = self.conv.weight
        if self._w_cache is None or self._w_version != w._version:
            OC, IC, KH, KW = w.shape
            # [OC, IC, KH, KW] -> [KH, KW, IC, OC]
            w_t = w.detach().permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC)
            self._w_cache = w_t
            self._w_version = w._version
        return self._w_cache

    def forward(self, x):
        x = x.cuda()
        w_cache = self._get_weight_cache()
        return conv2d_relu_bias_fused(x, self.conv.weight, self.conv.bias,
                                       self.bias, None, w_cache)