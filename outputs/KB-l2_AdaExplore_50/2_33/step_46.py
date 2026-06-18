import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def bn_stats_kernel(
    X_ptr,
    scale_ptr,
    mean_ptr, var_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)

    s = tl.load(scale_ptr + pid_n)

    sum_y = tl.zeros((), dtype=tl.float32)
    sum_y2 = tl.zeros((), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        m_idx = m_start + offs_m
        mask = m_idx < M
        x = tl.load(X_ptr + m_idx * N + pid_n, mask=mask, other=0.0)
        y = x * s
        sum_y += tl.sum(y, axis=0)
        sum_y2 += tl.sum(y * y, axis=0)
    mean = sum_y / M
    var = sum_y2 / M - mean * mean
    tl.store(mean_ptr + pid_n, mean)
    tl.store(var_ptr + pid_n, var)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=4, num_stages=3),
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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_scale_bn_kernel(
    A_ptr, B_ptr, Y_ptr,
    gbias_ptr, scale_ptr,
    mean_ptr, var_ptr, weight_ptr, bn_bias_ptr,
    M, N, K, eps,
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
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_am = offs_am < M
    mask_bn = offs_bn < N

    for k0 in range(0, K, BLOCK_K):
        k_remaining = K - k0
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < k_remaining) & mask_am[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & mask_bn[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision='tf32')
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add linear bias
    gb = tl.load(gbias_ptr + offs_bn, mask=mask_bn, other=0.0)
    acc = acc + gb[None, :]

    # Apply scale * BN affine in epilogue
    s = tl.load(scale_ptr + offs_bn, mask=mask_bn, other=0.0)
    mean = tl.load(mean_ptr + offs_bn, mask=mask_bn, other=0.0)
    var = tl.load(var_ptr + offs_bn, mask=mask_bn, other=0.0)
    w = tl.load(weight_ptr + offs_bn, mask=mask_bn, other=0.0)
    bnb = tl.load(bn_bias_ptr + offs_bn, mask=mask_bn, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    a_coef = s * w * inv_std
    b_coef = bnb - mean * w * inv_std

    y = acc * a_coef[None, :] + b_coef[None, :]

    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = Y_ptr + offs_ym[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    mask_y = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, y, mask=mask_y)


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
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features

        if self.training:
            # Need stats over (linear_out * scale) — do GEMM first, then stats, then fused apply.
            prev_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            out = torch.addmm(self.gemm.bias, x, self.gemm.weight.t())
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

            mean = torch.empty(N, device=x.device, dtype=torch.float32)
            var = torch.empty(N, device=x.device, dtype=torch.float32)
            bn_stats_kernel[(N,)](
                out, self.scale,
                mean, var,
                M, N,
                BLOCK_M=1024,
                num_warps=8,
            )
            with torch.no_grad():
                self.bn.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                unbiased_var = var * (M / max(M - 1, 1))
                self.bn.running_var.mul_(1 - self.momentum).add_(unbiased_var, alpha=self.momentum)
            use_mean = mean
            use_var = var

            y = torch.empty_like(out)
            grid_bn = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            fused_scale_bn_apply_kernel[grid_bn](
                out, y,
                self.scale, use_mean, use_var, self.bn.weight, self.bn.bias,
                M, N, self.eps,
            )
            return y
        else:
            # Eval mode: fuse GEMM + bias + scale + BN affine into one kernel.
            use_mean = self.bn.running_mean
            use_var = self.bn.running_var

            # Weight: nn.Linear stores weight as (out, in). We want B such that A @ B = out.
            # A = x (M, K), B = weight.t() (K, N).
            B = self.gemm.weight  # (N, K)
            y = torch.empty((M, N), device=x.device, dtype=x.dtype)

            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
            fused_gemm_scale_bn_kernel[grid](
                x, B, y,
                self.gemm.bias, self.scale,
                use_mean, use_var, self.bn.weight, self.bn.bias,
                M, N, K, self.eps,
                x.stride(0), x.stride(1),
                B.stride(1), B.stride(0),  # B is (N,K), we use it as (K,N) via swapped strides
                y.stride(0), y.stride(1),
            )
            return y