import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OHW', 'K_TOTAL'],
)
@triton.jit
def conv_fused_kernel(
    x_ptr,         # [N, IC, IH, IW]
    w_ptr,         # [OC, IC, KH, KW]
    cb_ptr,        # [OC] conv bias
    eb_ptr,        # [OC] extra bias
    out_ptr,       # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    OHW, K_TOTAL,
    constant_value, scaling_factor,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)            # batch
    pid_m = tl.program_id(1)            # OHW tile
    pid_oc = tl.program_id(2)           # OC tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # output spatial idx
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # OC idx

    m_mask = offs_m < OHW
    n_mask = offs_n < OC

    oh = offs_m // OW
    ow = offs_m % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K dimension: IC * KH * KW
    KHKW = KH * KW

    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_TOTAL

        ic = offs_k // KHKW
        rem = offs_k % KHKW
        kh = rem // KW
        kw = rem % KW

        # Load X: [BLOCK_M, BLOCK_K]
        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw[None, :]
        ic_b = ic[None, :]

        x_offset = ((pid_n * IC + ic_b) * IH + ih) * IW + iw
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        # Load W: [BLOCK_K, BLOCK_N], weight layout [OC, IC, KH, KW]
        w_offset = (offs_n[None, :] * IC + ic[:, None]) * KHKW + rem[:, None]
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # Add conv bias
    cb = tl.load(cb_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + cb[None, :]

    # min, +extra bias, *scale
    acc = tl.minimum(acc, constant_value)
    eb = tl.load(eb_ptr + offs_n, mask=n_mask, other=0.0)
    acc = (acc + eb[None, :]) * scaling_factor

    # Store: out[n, oc, oh, ow]
    out_offset = ((pid_n * OC + offs_n[None, :]) * OH + oh[:, None]) * OW + ow[:, None]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


def conv_fused(x, weight, conv_bias, extra_bias, constant_value, scaling_factor):
    x = x.contiguous()
    weight = weight.contiguous()
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1
    OHW = OH * OW
    K_TOTAL = IC * KH * KW

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    cb = conv_bias.contiguous().view(-1)
    eb = extra_bias.contiguous().view(-1)

    grid = lambda meta: (
        N,
        triton.cdiv(OHW, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    conv_fused_kernel[grid](
        x, weight, cb, eb, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        OHW, K_TOTAL,
        float(constant_value), float(scaling_factor),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return conv_fused(
            x,
            self.conv.weight,
            self.conv.bias,
            self.bias,
            self.constant_value,
            self.scaling_factor,
        )