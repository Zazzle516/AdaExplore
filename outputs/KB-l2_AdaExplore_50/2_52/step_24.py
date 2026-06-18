import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_mish_bn_kernel(
    a_ptr,  # [M, K]  im2col matrix
    b_ptr,  # [K, N]  weight reshaped (C_in*KH*KW, C_out)
    bias_ptr,  # [N]
    scale_ptr,  # [N]
    shift_ptr,  # [N]
    c_ptr,  # [M, N] then we permute on the host? we'll write directly to [N, C_out, H_out, W_out] via host permute
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # mish
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = acc * th

    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    shift = tl.load(shift_ptr + offs_n, mask=mask_n, other=0.0)
    out = y * scale[None, :] + shift[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'K', 'KH', 'KW'],
)
@triton.jit
def im2col_kernel(
    x_ptr,  # [N, C_in, H_in, W_in]
    out_ptr,  # [M, K] where M = N*H_out*W_out, K = C_in*KH*KW
    N, C_in, H_in, W_in,
    H_out, W_out, KH, KW,
    M, K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_k = offs_k < K

    HW_out = H_out * W_out
    n = offs_m // HW_out
    rem_m = offs_m % HW_out
    oh = rem_m // W_out
    ow = rem_m % W_out

    KHW = KH * KW
    ic = offs_k // KHW
    rem_k = offs_k % KHW
    kh = rem_k // KW
    kw = rem_k % KW

    ih = oh[:, None] + kh[None, :]
    iw = ow[:, None] + kw[None, :]
    x_off = (
        n[:, None] * (C_in * H_in * W_in)
        + ic[None, :] * (H_in * W_in)
        + ih * W_in
        + iw
    )
    mask = mask_m[:, None] & mask_k[None, :]
    vals = tl.load(x_ptr + x_off, mask=mask, other=0.0)

    out_off = offs_m[:, None] * K + offs_k[None, :]
    tl.store(out_ptr + out_off, vals, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        if self.training:
            x = self.conv(x)
            x = torch.multiply(torch.tanh(F.softplus(x)), x)
            x = self.bn(x)
            return x

        x = x.contiguous()
        N, C_in, H_in, W_in = x.shape
        C_out = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1
        HW_out = H_out * W_out
        M = N * HW_out
        K = C_in * KH * KW

        weight = self.conv.weight.contiguous()  # [C_out, C_in, KH, KW]
        # reshape to [K, C_out]
        w_mat = weight.view(C_out, K).t().contiguous()  # [K, C_out]

        bias = self.conv.bias.contiguous() if self.conv.bias is not None else torch.zeros(C_out, device=x.device, dtype=x.dtype)

        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias
        invstd = torch.rsqrt(running_var + self.eps)
        scale = (bn_weight * invstd).contiguous()
        shift = (bn_bias - running_mean * scale).contiguous()

        # im2col workspace [M, K]
        a = torch.empty((M, K), device=x.device, dtype=x.dtype)
        grid_im = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(K, meta['BLOCK_K']))
        im2col_kernel[grid_im](
            x, a,
            N, C_in, H_in, W_in,
            H_out, W_out, KH, KW,
            M, K,
        )

        # GEMM output [M, C_out]
        c = torch.empty((M, C_out), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
        gemm_mish_bn_kernel[grid](
            a, w_mat, bias, scale, shift, c,
            M, C_out, K,
            a.stride(0), a.stride(1),
            w_mat.stride(0), w_mat.stride(1),
            c.stride(0), c.stride(1),
        )

        # reshape [N*H_out*W_out, C_out] -> [N, H_out, W_out, C_out] -> [N, C_out, H_out, W_out]
        out = c.view(N, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()
        return out