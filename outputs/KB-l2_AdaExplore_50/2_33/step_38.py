import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_scale_bias_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def col_stats_kernel(
    Z_ptr, mean_ptr, invstd_ptr,
    M, N, eps,
    stride_m, stride_n,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)  # column index
    if pid >= N:
        return

    offs_m = tl.arange(0, BLOCK_M)
    sum_ = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sumsq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        idx = m_start + offs_m
        mask = idx < M
        ptrs = Z_ptr + idx * stride_m + pid * stride_n
        x = tl.load(ptrs, mask=mask, other=0.0)
        sum_ += x
        sumsq += x * x

    s = tl.sum(sum_, axis=0)
    sq = tl.sum(sumsq, axis=0)
    mean = s / M
    var = sq / M - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + pid, mean)
    tl.store(invstd_ptr + pid, invstd)


@triton.jit
def bn_apply_kernel(
    Z_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr, Out_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mean = tl.load(mean_ptr + offs_n, mask=offs_n < N, other=0.0)
    invstd = tl.load(invstd_ptr + offs_n, mask=offs_n < N, other=0.0)
    w = tl.load(weight_ptr + offs_n, mask=offs_n < N, other=0.0)
    b = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)

    ptrs = Z_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    out_ptrs = Out_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(ptrs, mask=mask, other=0.0)
    y = (x - mean[None, :]) * invstd[None, :] * w[None, :] + b[None, :]
    tl.store(out_ptrs, y, mask=mask)


@triton.jit
def bn_eval_kernel(
    Z_ptr, run_mean_ptr, run_var_ptr, weight_ptr, bias_ptr, Out_ptr,
    M, N, eps,
    stride_m, stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rm = tl.load(run_mean_ptr + offs_n, mask=offs_n < N, other=0.0)
    rv = tl.load(run_var_ptr + offs_n, mask=offs_n < N, other=0.0)
    w = tl.load(weight_ptr + offs_n, mask=offs_n < N, other=0.0)
    b = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    invstd = 1.0 / tl.sqrt(rv + eps)

    ptrs = Z_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    out_ptrs = Out_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(ptrs, mask=mask, other=0.0)
    y = (x - rm[None, :]) * invstd[None, :] * w[None, :] + b[None, :]
    tl.store(out_ptrs, y, mask=mask)


def gemm_scale_bias(x, weight, bias, scale):
    M, K = x.shape
    N, _ = weight.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    wt = weight.t().contiguous()
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_scale_bias_kernel[grid](
        x, wt, bias, scale, out,
        M, N, K,
        x.stride(0), x.stride(1),
        wt.stride(0), wt.stride(1),
        out.stride(0), out.stride(1),
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
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        weight = self.gemm.weight
        bias = self.gemm.bias
        scale = self.scale.view(-1)

        # Fused GEMM + bias + scale
        z = gemm_scale_bias(x, weight, bias, scale)

        M, N = z.shape

        if self.training:
            mean = torch.empty((N,), device=z.device, dtype=torch.float32)
            invstd = torch.empty((N,), device=z.device, dtype=torch.float32)
            BLOCK_M = 256
            grid = (N,)
            col_stats_kernel[grid](
                z, mean, invstd,
                M, N, self.eps,
                z.stride(0), z.stride(1),
                BLOCK_M=BLOCK_M,
            )
            # update running stats
            with torch.no_grad():
                var_unbiased = (1.0 / (invstd * invstd) - self.eps) * (M / max(M - 1, 1))
                var_biased = 1.0 / (invstd * invstd) - self.eps
                self.bn.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                self.bn.running_var.mul_(1 - self.momentum).add_(var_unbiased, alpha=self.momentum)
                self.bn.num_batches_tracked.add_(1)

            out = torch.empty_like(z)
            BLOCK_M2 = 64
            BLOCK_N2 = 128
            grid2 = (triton.cdiv(M, BLOCK_M2), triton.cdiv(N, BLOCK_N2))
            bn_apply_kernel[grid2](
                z, mean, invstd, self.bn.weight, self.bn.bias, out,
                M, N,
                z.stride(0), z.stride(1),
                BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2,
            )
            return out
        else:
            out = torch.empty_like(z)
            BLOCK_M2 = 64
            BLOCK_N2 = 128
            grid2 = (triton.cdiv(M, BLOCK_M2), triton.cdiv(N, BLOCK_N2))
            bn_eval_kernel[grid2](
                z, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, out,
                M, N, self.eps,
                z.stride(0), z.stride(1),
                BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2,
            )
            return out