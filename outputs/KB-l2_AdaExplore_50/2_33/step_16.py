import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_scale_stats_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr, C_ptr,
    sum_ptr, sumsq_ptr,
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
        a_mask = mask_m[:, None] & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask)

    # column-wise partial stats: reduce along M-rows of this tile
    acc_masked = tl.where(mask, acc, 0.0)
    col_sum = tl.sum(acc_masked, axis=0)
    col_sumsq = tl.sum(acc_masked * acc_masked, axis=0)

    tl.atomic_add(sum_ptr + offs_n, col_sum, mask=mask_n)
    tl.atomic_add(sumsq_ptr + offs_n, col_sumsq, mask=mask_n)


@triton.jit
def bn_apply_kernel(
    X_ptr, Y_ptr,
    sum_ptr, sumsq_ptr, weight_ptr, bias_ptr,
    M, N,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    s = tl.load(sum_ptr + offs_n, mask=mask_n, other=0.0)
    ss = tl.load(sumsq_ptr + offs_n, mask=mask_n, other=0.0)
    w = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0)
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)

    mean = s / M
    var = ss / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = w * inv_std
    shift = b - mean * scale

    ptrs = offs_m[:, None] * N + offs_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    x = tl.load(X_ptr + ptrs, mask=mask, other=0.0)
    y = x * scale[None, :] + shift[None, :]
    tl.store(Y_ptr + ptrs, y, mask=mask)


@triton.jit
def bn_stats_kernel(
    X_ptr,
    sum_ptr, sumsq_ptr,
    M, N,
    stride_xm, stride_xn,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)

    sum_x = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        m_idx = m_start + offs_m
        mask = m_idx < M
        x = tl.load(X_ptr + m_idx * stride_xm + pid_n * stride_xn, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    tl.store(sum_ptr + pid_n, sum_x)
    tl.store(sumsq_ptr + pid_n, sum_sq)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum

        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features

        W = self.gemm.weight  # (N, K)
        bias = self.gemm.bias
        scale = self.scale

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        sum_buf = torch.zeros(N, device=x.device, dtype=torch.float32)
        sumsq_buf = torch.zeros(N, device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        gemm_scale_stats_kernel[grid](
            x, W, bias, scale, out,
            sum_buf, sumsq_buf,
            M, N, K,
            x.stride(0), x.stride(1),
            1, K,
            out.stride(0), out.stride(1),
        )

        if self.training:
            with torch.no_grad():
                mean = sum_buf / M
                var = sumsq_buf / M - mean * mean
                self.bn.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                unbiased_var = var * (M / max(M - 1, 1))
                self.bn.running_var.mul_(1 - self.momentum).add_(unbiased_var, alpha=self.momentum)
            use_sum = sum_buf
            use_sumsq = sumsq_buf
            denom_M = M
        else:
            # Build effective sum/sumsq from running stats so the same apply-kernel works
            use_sum = self.bn.running_mean * M
            use_sumsq = (self.bn.running_var + self.bn.running_mean * self.bn.running_mean) * M
            denom_M = M

        y = torch.empty_like(out)
        BLOCK_M_APPLY = 64
        BLOCK_N_APPLY = 256
        grid_apply = (triton.cdiv(M, BLOCK_M_APPLY), triton.cdiv(N, BLOCK_N_APPLY))
        bn_apply_kernel[grid_apply](
            out, y,
            use_sum, use_sumsq, self.bn.weight, self.bn.bias,
            denom_M, N,
            eps=float(self.eps),
            BLOCK_M=BLOCK_M_APPLY,
            BLOCK_N=BLOCK_N_APPLY,
        )
        return y