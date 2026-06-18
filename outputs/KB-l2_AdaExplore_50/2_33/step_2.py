import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_scale_kernel(
    X_ptr, W_ptr, B_ptr, S_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
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
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    s = tl.load(S_ptr + offs_n, mask=mask_n, other=0.0)
    acc = (acc + b[None, :]) * s[None, :]

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1024}, num_warps=4),
        triton.Config({'BLOCK_M': 2048}, num_warps=8),
        triton.Config({'BLOCK_M': 4096}, num_warps=8),
        triton.Config({'BLOCK_M': 512}, num_warps=4),
    ],
    key=['M'],
)
@triton.jit
def col_stats_kernel(
    Y_ptr, MEAN_ptr, VAR_ptr,
    M, N,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    sum_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sumsq_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        cur = m_start + offs_m
        mask = cur < M
        y = tl.load(Y_ptr + cur * stride_ym + pid_n * stride_yn, mask=mask, other=0.0)
        sum_acc += tl.where(mask, y, 0.0)
        sumsq_acc += tl.where(mask, y * y, 0.0)
    s = tl.sum(sum_acc, axis=0)
    sq = tl.sum(sumsq_acc, axis=0)
    mean = s / M
    var = sq / M - mean * mean
    tl.store(MEAN_ptr + pid_n, mean)
    tl.store(VAR_ptr + pid_n, var)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def bn_apply_kernel(
    Y_ptr, MEAN_ptr, VAR_ptr, GAMMA_ptr, BETA_ptr, OUT_ptr,
    M, N, eps,
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
    ptrs = offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y = tl.load(Y_ptr + ptrs, mask=mask, other=0.0)
    mean = tl.load(MEAN_ptr + offs_n, mask=mask_n, other=0.0)
    var = tl.load(VAR_ptr + offs_n, mask=mask_n, other=0.0)
    gamma = tl.load(GAMMA_ptr + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(BETA_ptr + offs_n, mask=mask_n, other=0.0)
    inv = 1.0 / tl.sqrt(var + eps)
    out = (y - mean[None, :]) * inv[None, :] * gamma[None, :] + beta[None, :]
    tl.store(OUT_ptr + ptrs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_scale_bn_eval_kernel(
    X_ptr, W_ptr, B_ptr, S_ptr, CS_ptr, CSH_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
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
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    cs = tl.load(CS_ptr + offs_n, mask=mask_n, other=0.0)
    csh = tl.load(CSH_ptr + offs_n, mask=mask_n, other=0.0)
    # (acc + b) * cs + csh   where cs = s*gamma/sqrt(var+eps), csh = beta - running_mean*gamma/sqrt(var+eps)
    out = (acc + b[None, :]) * cs[None, :] + csh[None, :]

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


# Eval kernel requires BLOCK_K and GROUP_M as constexpr; override autotune set
gemm_scale_bn_eval_kernel = triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)(gemm_scale_bn_eval_kernel.fn)


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
        W = self.gemm.weight  # [N, K]
        B = self.gemm.bias    # [N]
        S = self.scale        # [N]
        gamma = self.bn.weight
        beta = self.bn.bias

        if self.training:
            y = torch.empty((M, N), device=x.device, dtype=x.dtype)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
            gemm_scale_kernel[grid](
                x, W, B, S, y,
                M, N, K,
                x.stride(0), x.stride(1),
                W.stride(0), W.stride(1),
                y.stride(0), y.stride(1),
            )
            mean = torch.empty(N, device=x.device, dtype=torch.float32)
            var = torch.empty(N, device=x.device, dtype=torch.float32)
            grid2 = (N,)
            col_stats_kernel[grid2](
                y, mean, var,
                M, N,
                y.stride(0), y.stride(1),
            )
            # update running stats
            with torch.no_grad():
                unbiased_var = var * (M / max(M - 1, 1))
                self.bn.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                self.bn.running_var.mul_(1 - self.momentum).add_(unbiased_var, alpha=self.momentum)
                self.bn.num_batches_tracked.add_(1)

            out = torch.empty_like(y)
            grid3 = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            bn_apply_kernel[grid3](
                y, mean, var, gamma, beta, out,
                M, N, self.eps,
                y.stride(0), y.stride(1),
            )
            return out
        else:
            running_mean = self.bn.running_mean
            running_var = self.bn.running_var
            inv = torch.rsqrt(running_var + self.eps)
            combined_scale = S * gamma * inv          # [N]
            combined_shift = beta - running_mean * gamma * inv  # [N]
            out = torch.empty((M, N), device=x.device, dtype=x.dtype)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
            gemm_scale_bn_eval_kernel[grid](
                x, W, B, S, combined_scale, combined_shift, out,
                M, N, K,
                x.stride(0), x.stride(1),
                W.stride(0), W.stride(1),
                out.stride(0), out.stride(1),
            )
            return out