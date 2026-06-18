import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def bn_stats_kernel(
    X_ptr,
    scale_ptr, gemm_bias_ptr,
    mean_ptr, var_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
):
    # X here is the raw gemm output (no scale applied yet).
    # We compute stats of (X * scale) without materializing it.
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)

    s = tl.load(scale_ptr + pid_n)

    # First pass: mean of (x * s)
    sum_x = tl.zeros((), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        m_idx = m_start + offs_m
        mask = m_idx < M
        x = tl.load(X_ptr + m_idx * N + pid_n, mask=mask, other=0.0)
        sum_x += tl.sum(x, axis=0)
    mean = (sum_x * s) / M

    # Second pass: variance
    sum_d2 = tl.zeros((), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        m_idx = m_start + offs_m
        mask = m_idx < M
        x = tl.load(X_ptr + m_idx * N + pid_n, mask=mask, other=0.0)
        d = tl.where(mask, x * s - mean, 0.0)
        sum_d2 += tl.sum(d * d, axis=0)
    var = sum_d2 / M
    tl.store(mean_ptr + pid_n, mean)
    tl.store(var_ptr + pid_n, var)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_scale_bn_apply_kernel(
    X_ptr, Y_ptr,
    scale_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    M, N, eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    s = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    var = tl.load(var_ptr + offs_n, mask=mask_n, other=0.0)
    w = tl.load(weight_ptr + offs_n, mask=mask_n, other=0.0)
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    a_coef = s * w * inv_std
    b_coef = b - mean * w * inv_std

    ptrs = offs_m[:, None] * N + offs_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    x = tl.load(X_ptr + ptrs, mask=mask, other=0.0)
    y = x * a_coef[None, :] + b_coef[None, :]
    tl.store(Y_ptr + ptrs, y, mask=mask)


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

        # GEMM via cuBLAS (heaviest op, executed at runtime).
        # Use addmm: out = bias + x @ W^T
        out = torch.addmm(self.gemm.bias, x, self.gemm.weight.t())

        # BatchNorm1d stats over (out * scale)
        if self.training:
            mean = torch.empty(N, device=x.device, dtype=torch.float32)
            var = torch.empty(N, device=x.device, dtype=torch.float32)
            bn_stats_kernel[(N,)](
                out, self.scale, self.gemm.bias,
                mean, var,
                M, N,
                BLOCK_M=1024,
                num_warps=4,
            )
            with torch.no_grad():
                self.bn.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                unbiased_var = var * (M / max(M - 1, 1))
                self.bn.running_var.mul_(1 - self.momentum).add_(unbiased_var, alpha=self.momentum)
            use_mean = mean
            use_var = var
        else:
            use_mean = self.bn.running_mean
            use_var = self.bn.running_var

        y = torch.empty_like(out)
        grid_bn = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        fused_scale_bn_apply_kernel[grid_bn](
            out, y,
            self.scale, use_mean, use_var, self.bn.weight, self.bn.bias,
            M, N, self.eps,
        )
        return y