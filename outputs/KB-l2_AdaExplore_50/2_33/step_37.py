import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_scale_bias_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # bias + scale
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    scale = tl.load(scale_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Column-wise reduction: compute sum and sum-of-squares per column
@triton.jit
def col_reduce_kernel(
    X_ptr, sum_ptr, sqsum_ptr,
    M, N,
    stride_xm, stride_xn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)

    mask_n = offs_n < N
    ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    mask = (offs_m[:, None] < M) & mask_n[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    sq = tl.sum(x * x, axis=0)

    tl.store(sum_ptr + offs_n, s, mask=mask_n)
    tl.store(sqsum_ptr + offs_n, sq, mask=mask_n)


# Apply BN given mean/invstd (and update running stats handled outside)
@triton.jit
def bn_apply_kernel(
    X_ptr, Y_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    invstd = tl.load(invstd_ptr + offs_n, mask=mask_n, other=0.0)
    w = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0)
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)

    y = (x - mean[None, :]) * invstd[None, :] * w[None, :] + b[None, :]

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=mask)


# Eval path: fused GEMM + scale + bias + BN-apply (folded) in epilogue
@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_scale_bn_eval_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr,
    rmean_ptr, rinvstd_ptr, bnw_ptr, bnb_ptr,
    C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_cn < N
    bias = tl.load(bias_ptr + offs_cn, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_cn, mask=mask_n, other=0.0)
    rmean = tl.load(rmean_ptr + offs_cn, mask=mask_n, other=0.0)
    rinvstd = tl.load(rinvstd_ptr + offs_cn, mask=mask_n, other=0.0)
    bnw = tl.load(bnw_ptr + offs_cn, mask=mask_n, other=0.0)
    bnb = tl.load(bnb_ptr + offs_cn, mask=mask_n, other=0.0)

    z = (acc + bias[None, :]) * scale[None, :]
    y = (z - rmean[None, :]) * rinvstd[None, :] * bnw[None, :] + bnb[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & mask_n[None, :]
    tl.store(c_ptrs, y, mask=c_mask)


def triton_gemm_scale_bias(x, weight, bias, scale):
    # x: (M, K), weight: (N, K) -> compute x @ weight.T = (M, N), then + bias, * scale
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_scale_bias_kernel[grid](
        x, weight, bias, scale, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # transposed
        out.stride(0), out.stride(1),
    )
    return out


def triton_gemm_scale_bn_eval(x, weight, bias, scale, rmean, rinvstd, bnw, bnb):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_scale_bn_eval_kernel[grid](
        x, weight, bias, scale,
        rmean, rinvstd, bnw, bnb,
        out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


def triton_bn_apply(x, mean, invstd, weight, bias):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_M = 64
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    bn_apply_kernel[grid](
        x, out, mean, invstd, weight, bias,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


def triton_col_reduce(x):
    M, N = x.shape
    sum_ = torch.empty(N, device=x.device, dtype=torch.float32)
    sqsum = torch.empty(N, device=x.device, dtype=torch.float32)
    # M=1024, pick BLOCK_M = next power of 2 of M
    BLOCK_M = triton.next_power_of_2(M)
    BLOCK_N = 128
    grid = (triton.cdiv(N, BLOCK_N),)
    col_reduce_kernel[grid](
        x, sum_, sqsum,
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return sum_, sqsum


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self.momentum = momentum
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        weight = self.gemm.weight.contiguous()
        bias = self.gemm.bias.contiguous()
        scale = self.scale.contiguous()

        if not self.training:
            # Fold BN into single fused kernel
            rmean = self.bn.running_mean
            rvar = self.bn.running_var
            rinvstd = torch.rsqrt(rvar + self.eps)
            bnw = self.bn.weight
            bnb = self.bn.bias
            y = triton_gemm_scale_bn_eval(x, weight, bias, scale, rmean, rinvstd, bnw, bnb)
            return y
        else:
            # GEMM + scale + bias fused
            z = triton_gemm_scale_bias(x, weight, bias, scale)
            M = z.shape[0]
            # Compute mean/var via column reduction
            sum_, sqsum = triton_col_reduce(z)
            mean = sum_ / M
            var = sqsum / M - mean * mean
            invstd = torch.rsqrt(var + self.eps)
            # Update running stats (unbiased var for running_var)
            with torch.no_grad():
                unbiased_var = var * (M / max(M - 1, 1))
                self.bn.running_mean.mul_(1 - self.momentum).add_(mean.detach(), alpha=self.momentum)
                self.bn.running_var.mul_(1 - self.momentum).add_(unbiased_var.detach(), alpha=self.momentum)
            y = triton_bn_apply(z, mean, invstd, self.bn.weight, self.bn.bias)
            return y