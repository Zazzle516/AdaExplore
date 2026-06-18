import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_relu_bias_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # OC tile (M = OC)
    pid_n = tl.program_id(1)  # spatial tile (N = OH*OW)
    pid_b = tl.program_id(2)  # batch index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC offsets
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial offsets

    oh = offs_n // OW
    ow = offs_n % OW

    K = IC * KH * KW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        ic = offs_k // (KH * KW)
        kh = (offs_k // KW) % KH
        kw = offs_k % KW

        # Load weight block [BLOCK_M, BLOCK_K]
        w_offs = (offs_m[:, None] * stride_wo +
                  ic[None, :] * stride_wi +
                  kh[None, :] * stride_wkh +
                  kw[None, :] * stride_wkw)
        w_mask = (offs_m[:, None] < OC) & k_mask[None, :]
        w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        # Load input block [BLOCK_K, BLOCK_N]
        ih = oh[None, :] + kh[:, None]
        iw = ow[None, :] + kw[:, None]
        x_offs = (pid_b * stride_xn +
                  ic[:, None] * stride_xc +
                  ih * stride_xh +
                  iw * stride_xw)
        x_mask = k_mask[:, None] & (offs_n[None, :] < OH * OW)
        x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

        acc += tl.dot(w_vals, x_vals)

    # Add conv bias
    b_vals = tl.load(b_ptr + offs_m, mask=offs_m < OC, other=0.0)
    acc = acc + b_vals[:, None]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Add extra bias (per output channel)
    extra_bias = tl.load(bias_ptr + offs_m, mask=offs_m < OC, other=0.0)
    acc = acc + extra_bias[:, None]

    # Store
    out_offs = (pid_b * stride_on +
                offs_m[:, None] * stride_oc +
                oh[None, :] * stride_oh +
                ow[None, :] * stride_ow)
    out_mask = (offs_m[:, None] < OC) & (offs_n[None, :] < OH * OW)
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv2d_relu_bias(x, weight, conv_bias, extra_bias):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    conv_bias = conv_bias.contiguous()
    extra_bias_flat = extra_bias.contiguous().view(-1)

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(OC, meta['BLOCK_M']),
        triton.cdiv(OH * OW, meta['BLOCK_N']),
        N,
    )

    conv2d_relu_bias_kernel[grid](
        x, weight, conv_bias, extra_bias_flat, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.cuda()
        return conv2d_relu_bias(x, self.conv.weight, self.conv.bias, self.bias)