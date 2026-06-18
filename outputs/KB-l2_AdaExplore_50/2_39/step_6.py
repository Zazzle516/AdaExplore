import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_scale_kernel(
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def col_stats_kernel(
    X_ptr, mean_ptr, var_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    sum_x = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        offs = m_start + offs_m
        mask = offs < M
        x = tl.load(X_ptr + offs * stride_m + pid_n * stride_n, mask=mask, other=0.0)
        sum_x += tl.where(mask, x, 0.0)
        sum_x2 += tl.where(mask, x * x, 0.0)
    s = tl.sum(sum_x, axis=0)
    s2 = tl.sum(sum_x2, axis=0)
    mean = s / M
    var = s2 / M - mean * mean
    tl.store(mean_ptr + pid_n, mean)
    tl.store(var_ptr + pid_n, var)


@triton.jit
def bn_apply_kernel(
    X_ptr, Y_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    M, N, eps,
    stride_m, stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    var = tl.load(var_ptr + offs_n, mask=mask_n, other=0.0)
    w = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0)
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    invstd = 1.0 / tl.sqrt(var + eps)
    scale = w * invstd
    shift = b - mean * scale

    ptrs = X_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    x = tl.load(ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    y = x * scale[None, :] + shift[None, :]
    out_ptrs = Y_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    tl.store(out_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features

        W = self.gemm.weight  # (N, K)
        bias = self.gemm.bias  # (N,)
        scale = self.scale.view(-1)  # (N,)

        gemm_out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # Use W^T as B: A (M,K) * B (K,N)
        Wt = W.t().contiguous()  # (K, N) contiguous

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        gemm_scale_kernel[grid](
            x, Wt, bias, scale, gemm_out,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            gemm_out.stride(0), gemm_out.stride(1),
        )

        weight = self.bn.weight
        bn_bias = self.bn.bias

        if self.training:
            # Compute batch stats
            mean = torch.empty((N,), device=x.device, dtype=torch.float32)
            var = torch.empty((N,), device=x.device, dtype=torch.float32)
            BLOCK_M_STAT = 1024
            col_stats_kernel[(N,)](
                gemm_out, mean, var,
                M, N,
                gemm_out.stride(0), gemm_out.stride(1),
                BLOCK_M=BLOCK_M_STAT,
                num_warps=8,
            )
            # Update running stats
            with torch.no_grad():
                momentum = self.bn.momentum
                unbiased_var = var * (M / (M - 1)) if M > 1 else var
                self.bn.running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
                self.bn.running_var.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)
                self.bn.num_batches_tracked.add_(1)
            use_mean = mean
            use_var = var
        else:
            use_mean = self.bn.running_mean
            use_var = self.bn.running_var

        out = torch.empty_like(gemm_out)
        BLOCK_M = 128
        BLOCK_N = 128
        grid2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        bn_apply_kernel[grid2](
            gemm_out, out, use_mean, use_var, weight, bn_bias,
            M, N, self.eps,
            gemm_out.stride(0), gemm_out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out