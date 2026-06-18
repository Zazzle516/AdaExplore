import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_scale_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
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
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # bias + scale
    offs_n_real = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + offs_n_real, mask=offs_n_real < N, other=0.0)
    scale = tl.load(scale_ptr + offs_n_real, mask=offs_n_real < N, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    offs_m_real = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_ptrs = Out_ptr + offs_m_real[:, None] * stride_om + offs_n_real[None, :] * stride_on
    mask = (offs_m_real[:, None] < M) & (offs_n_real[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask)


def gemm_scale(x, weight, bias, scale):
    # x: (M, K), weight: (N, K), bias: (N,), scale: (N,)
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_scale_kernel[grid](
        x, weight, bias, scale, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # weight is (N,K) but we treat as K x N: B[k,n] = weight[n,k]
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def bn_stats_kernel(
    X_ptr, mean_ptr, invstd_ptr,
    M, N, eps,
    BLOCK_M: tl.constexpr,
):
    n = tl.program_id(0)
    sum_x = 0.0
    sum_x2 = 0.0
    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask = offs_m < M
        x = tl.load(X_ptr + offs_m * N + n, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
    mean = sum_x / M
    var = sum_x2 / M - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + n, mean)
    tl.store(invstd_ptr + n, invstd)


@triton.jit
def bn_apply_kernel(
    X_ptr, Out_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    invstd = tl.load(invstd_ptr + offs_n, mask=mask_n, other=0.0)
    w = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0)
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)

    ptrs = offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(X_ptr + ptrs, mask=mask, other=0.0)
    y = (x - mean[None, :]) * invstd[None, :] * w[None, :] + b[None, :]
    tl.store(Out_ptr + ptrs, y, mask=mask)


def batchnorm_train(x, weight, bias, running_mean, running_var, eps, momentum):
    M, N = x.shape
    mean = torch.empty(N, device=x.device, dtype=torch.float32)
    invstd = torch.empty(N, device=x.device, dtype=torch.float32)
    BLOCK_M_STATS = 1024
    bn_stats_kernel[(N,)](x, mean, invstd, M, N, eps, BLOCK_M=BLOCK_M_STATS, num_warps=4)

    out = torch.empty_like(x)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    bn_apply_kernel[grid](x, out, mean, invstd, weight, bias, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4)

    # Update running stats
    with torch.no_grad():
        var = (1.0 / (invstd * invstd)) - eps
        # unbiased var for running_var: var * M / (M-1)
        unbiased_var = var * (M / max(M - 1, 1))
        running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
        running_var.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)
    return out


def batchnorm_eval(x, weight, bias, running_mean, running_var, eps):
    M, N = x.shape
    invstd = 1.0 / torch.sqrt(running_var + eps)
    out = torch.empty_like(x)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    bn_apply_kernel[grid](x, out, running_mean, invstd, weight, bias, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.gemm.weight.contiguous()
        bias = self.gemm.bias.contiguous()
        scale = self.scale.contiguous().view(-1)

        y = gemm_scale(x, weight, bias, scale)

        if self.training:
            out = batchnorm_train(y, self.bn.weight, self.bn.bias,
                                  self.bn.running_mean, self.bn.running_var,
                                  self.eps, self.momentum)
            self.bn.num_batches_tracked.add_(1)
        else:
            out = batchnorm_eval(y, self.bn.weight, self.bn.bias,
                                 self.bn.running_mean, self.bn.running_var, self.eps)
        return out