import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'M', 'K'],
)
@triton.jit
def conv_fused_kernel(
    x_ptr, w_ptr, cb_ptr, bias_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW, KH, KW,
    constant_value, scaling_factor,
    M, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # output spatial tile (over N*OH*OW)
    pid_n = tl.program_id(1)  # OC tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # decompose m into (n, oh, ow)
    OHW = OH * OW
    n_idx = offs_m // OHW
    rem = offs_m % OHW
    oh_idx = rem // OW
    ow_idx = rem % OW

    m_mask = offs_m < M
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHW = KH * KW
    K_total = IC * KHW

    for k_start in range(0, K_total, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K_total

        ic = k_idx // KHW
        kk = k_idx % KHW
        kh = kk // KW
        kw = kk % KW

        # input gather: x[n, ic, oh+kh, ow+kw]
        # shape [BLOCK_M, BLOCK_K]
        ih = oh_idx[:, None] + kh[None, :]
        iw = ow_idx[:, None] + kw[None, :]
        x_offsets = (n_idx[:, None] * stride_xn +
                     ic[None, :] * stride_xc +
                     ih * stride_xh +
                     iw * stride_xw)
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        # weight gather: w[oc, ic, kh, kw], shape [BLOCK_K, BLOCK_N]
        w_offsets = (offs_n[None, :] * stride_wo +
                     ic[:, None] * stride_wi +
                     kh[:, None] * stride_wkh +
                     kw[:, None] * stride_wkw)
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # add conv bias
    cb = tl.load(cb_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + cb[None, :]

    # min with constant
    acc = tl.minimum(acc, constant_value)

    # add bias (per oc)
    b = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b[None, :]

    # scale
    acc = acc * scaling_factor

    # store output [N, OC, OH, OW]
    out_offsets = (n_idx[:, None] * stride_on +
                   offs_n[None, :] * stride_oc +
                   oh_idx[:, None] * stride_oh +
                   ow_idx[:, None] * stride_ow)
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


def fused_conv(x, weight, conv_bias, bias, constant_value, scaling_factor):
    x = x.contiguous()
    weight = weight.contiguous()
    N, IC, H, W = x.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    bias_flat = bias.contiguous().view(-1)
    cb_flat = conv_bias.contiguous().view(-1)

    M = N * OH * OW
    K = IC * KH * KW

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    conv_fused_kernel[grid](
        x, weight, cb_flat, bias_flat, out,
        N, IC, H, W,
        OC, OH, OW, KH, KW,
        float(constant_value), float(scaling_factor),
        M, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
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
        return fused_conv(
            x, self.conv.weight, self.conv.bias,
            self.bias, self.constant_value, self.scaling_factor,
        )