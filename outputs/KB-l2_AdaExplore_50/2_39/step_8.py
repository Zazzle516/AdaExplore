import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# GEMM with fused bias + scale, two outputs: y and y^2 partial sum we'll do separately
# Strategy:
#   Kernel 1: GEMM (x @ W^T + bias) * scale -> Y  [M,N], also during eval BN is constant so fold.
#   But BN at train uses batch stats. Model uses default train mode? Let's handle both.
# Since the model is constructed with default mode (train()), bn uses batch stats.
# We do:
#   1) GEMM kernel: writes Y = (x@W^T + b) * scale, fp32
#   2) Stats kernel: column-wise mean and var over M
#   3) Apply kernel: out = (Y - mean) * invstd * gamma + beta
# Also update running stats (in train mode).

GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_scale_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_am < M
    mask_n = offs_bn < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remain = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remain), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remain) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_bn, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_bn, mask=mask_n, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    y_ptrs = Y_ptr + offs_am[:, None] * stride_ym + offs_bn[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# Column-wise stats: each program handles one column (or a tile of columns)
@triton.jit
def col_stats_kernel(
    Y_ptr, mean_ptr, invstd_ptr,
    M, N, eps,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    sum_x = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        x = tl.load(ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / M
    var = sum_x2 / M - mean * mean
    var = tl.maximum(var, 0.0)
    invstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + offs_n, mean, mask=mask_n)
    tl.store(invstd_ptr + offs_n, invstd, mask=mask_n)


@triton.jit
def bn_apply_kernel(
    Y_ptr, out_ptr, mean_ptr, invstd_ptr, gamma_ptr, beta_ptr,
    M, N,
    stride_ym, stride_yn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    invstd = tl.load(invstd_ptr + offs_n, mask=mask_n, other=0.0)
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0)

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    x = tl.load(y_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    scale = invstd * gamma
    shift = beta - mean * scale
    out = x * scale[None, :] + shift[None, :]

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


# Eval-mode fused: GEMM + scale + BN affine in epilogue (running stats are constants)
@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bn_eval_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr,
    bn_scale_ptr, bn_shift_ptr,  # precomputed
    Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_am < M
    mask_n = offs_bn < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remain = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remain), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remain) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_bn, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_bn, mask=mask_n, other=0.0)
    bn_s = tl.load(bn_scale_ptr + offs_bn, mask=mask_n, other=0.0)
    bn_b = tl.load(bn_shift_ptr + offs_bn, mask=mask_n, other=0.0)

    out = (acc + bias[None, :]) * scale[None, :]
    out = out * bn_s[None, :] + bn_b[None, :]

    y_ptrs = Y_ptr + offs_am[:, None] * stride_ym + offs_bn[None, :] * stride_yn
    tl.store(y_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


def triton_gemm_scale(x, w, b, scale):
    M, K = x.shape
    N, _ = w.shape
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_scale_kernel[grid](
        x, w, b, scale, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),  # we treat w as (K, N) by swapping strides since w is (N, K)
        y.stride(0), y.stride(1),
    )
    return y


def triton_gemm_bn_eval(x, w, b, scale, bn_scale, bn_shift):
    M, K = x.shape
    N, _ = w.shape
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bn_eval_kernel[grid](
        x, w, b, scale, bn_scale, bn_shift, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),
        y.stride(0), y.stride(1),
    )
    return y


def compute_stats(y, eps):
    M, N = y.shape
    mean = torch.empty(N, device=y.device, dtype=torch.float32)
    invstd = torch.empty(N, device=y.device, dtype=torch.float32)
    BLOCK_N = 64
    BLOCK_M = 1024
    grid = (triton.cdiv(N, BLOCK_N),)
    col_stats_kernel[grid](
        y, mean, invstd,
        M, N, eps,
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return mean, invstd


def apply_bn(y, mean, invstd, gamma, beta):
    M, N = y.shape
    out = torch.empty_like(y)
    BLOCK_M = 64
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    bn_apply_kernel[grid](
        y, out, mean, invstd, gamma, beta,
        M, N,
        y.stride(0), y.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self.momentum = momentum
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.float()

        W = self.gemm.weight  # (N, K)
        b = self.gemm.bias    # (N,)
        scale = self.scale.view(-1)  # (N,)

        if not self.training:
            running_mean = self.bn.running_mean
            running_var = self.bn.running_var
            gamma = self.bn.weight
            beta = self.bn.bias
            invstd = torch.rsqrt(running_var + self.eps)
            bn_scale = gamma * invstd
            bn_shift = beta - running_mean * bn_scale
            out = triton_gemm_bn_eval(x, W.contiguous(), b.contiguous(),
                                       scale.contiguous(),
                                       bn_scale.contiguous(), bn_shift.contiguous())
            return out
        else:
            y = triton_gemm_scale(x, W.contiguous(), b.contiguous(), scale.contiguous())
            mean, invstd = compute_stats(y, self.eps)

            # update running stats
            with torch.no_grad():
                M = y.shape[0]
                # var = 1/invstd^2 - eps
                var = (1.0 / (invstd * invstd)) - self.eps
                self.bn.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                # PyTorch uses unbiased var for running_var update
                unbiased_var = var * (M / max(M - 1, 1))
                self.bn.running_var.mul_(1 - self.momentum).add_(unbiased_var, alpha=self.momentum)

            out = apply_bn(y, mean, invstd, self.bn.weight.contiguous(), self.bn.bias.contiguous())
            return out