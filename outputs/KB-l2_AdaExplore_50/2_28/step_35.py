import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Note: InstanceNorm2d on a tensor of shape (B, C, 1, 1) where H*W = 1.
# The variance over a single element is 0, so output is (x - x) / sqrt(0 + eps) = 0.
# Therefore instance_norm output is all zeros, and the final result is 0*y + y*y... wait.
# Let's recompute: x_norm = 0, then x = x_norm + y = y, then x = x * y = y * y.
# Wait, but actually F.instance_norm with single element: mean=x, var=0, output=(x-x)/sqrt(eps)=0.
# Yes. So final = (0 + y) * y = y * y.
# But safety contract says we cannot do graph-level shortcuts. Every operator must execute.
# So we must actually run the linear (bmm) and instance_norm.

# We'll fuse: linear (matmul + bias) then instance_norm (which gives 0), then add y, then mul y.
# Since instance_norm of single element is 0, the result depends only on y.
# But we must run the linear. So compute linear, then in epilogue do instance_norm 
# (which produces 0 elementwise per element since H*W=1), then add y, mul y.

# Actually, let me re-read: x has shape (batch_size, out_features) after linear.
# x.unsqueeze(1).unsqueeze(1) -> (batch_size, 1, 1, out_features)
# InstanceNorm2d(out_features) expects (N, C, H, W) where C=out_features.
# So shape is (batch_size, 1, 1, out_features) - C=1, H=1, W=out_features.
# Wait, but InstanceNorm2d was created with out_features channels.
# That's a mismatch in channels. Let me check what pytorch does.
# InstanceNorm2d doesn't have learnable params by default (affine=False).
# It normalizes over H,W per (N,C). With shape (batch_size, 1, 1, out_features):
# N=batch_size, C=1, H=1, W=out_features. So normalize each (n, 0) slice over H*W=out_features.
# So it computes mean and var over out_features dimension for each sample.
# That makes more sense!

# So instance_norm normalizes x (shape batch, out_features) along the out_features dim.
# This is like LayerNorm without affine.

# Plan:
# 1. Matmul kernel: x @ W^T + b -> (batch_size, out_features)
# 2. Norm kernel: per row, compute mean/var, normalize, then add y, mul y, output.

# We'll keep these as two kernels for simplicity (norm needs full row).


# ============================================================
# Tiled GEMM kernel with bias
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
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

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = mask_m[:, None] & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # treating weight as (K, N) transposed
        out.stride(0), out.stride(1),
    )
    return out


# ============================================================
# Fused instance-norm + add y + mul y kernel
# Normalizes each row of x (shape M, N), then output = (norm + y) * y
# ============================================================
@triton.jit
def fused_norm_addmul_kernel(
    x_ptr, y_ptr, out_ptr,
    M, N,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)

    x_row_ptr = x_ptr + row * N
    y_row_ptr = y_ptr + row * N
    out_row_ptr = out_ptr + row * N

    # Pass 1: compute mean using Welford-style two-pass with centered sumsq
    sum_val = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
    mean = sum_val / N

    sum_sq = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        d = tl.where(mask, x - mean, 0.0)
        sum_sq += tl.sum(d * d, axis=0)
    var = sum_sq / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    # Pass 3: normalize, then add y, mul y
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = (x - mean) * rstd
        result = (normed + y) * y
        tl.store(out_row_ptr + cols, result, mask=mask)


def fused_norm_addmul(x, y, eps):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = 2048
    grid = (M,)
    fused_norm_addmul_kernel[grid](
        x, y, out,
        M, N,
        EPS=eps,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, y):
        x = x.contiguous()
        y = y.contiguous()
        weight = self.bmm.weight.contiguous()
        bias = self.bmm.bias.contiguous()

        # Linear: x @ W^T + b
        x = triton_linear(x, weight, bias)

        # Fused instance_norm (per-row over out_features dim) + add y + mul y
        out = fused_norm_addmul(x, y, self.eps)
        return out