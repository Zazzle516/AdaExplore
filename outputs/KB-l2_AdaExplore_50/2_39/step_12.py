import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_affine_kernel(
    A_ptr, B_ptr, A_eff_ptr, B_eff_ptr, OUT_ptr,
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

    # Load per-column affine: out = acc * a_eff + b_eff
    a_eff = tl.load(A_eff_ptr + offs_n, mask=mask_n, other=0.0)
    b_eff = tl.load(B_eff_ptr + offs_n, mask=mask_n, other=0.0)
    out = acc * a_eff[None, :] + b_eff[None, :]

    out_ptrs = OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, out, mask=(mask_m[:, None]) & (mask_n[None, :]))


# Training-mode: two-stage BN reduction
@triton.jit
def bn_partial_kernel(
    X_ptr,  # [M, N]
    PSUM_ptr,  # [num_row_tiles, N]
    PSUMSQ_ptr,  # [num_row_tiles, N]
    M, N,
    stride_xm, stride_xn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    s = tl.sum(x, axis=0)
    sq = tl.sum(x * x, axis=0)

    num_row_tiles = tl.num_programs(1)
    out_off = pid_m * N + offs_n
    tl.store(PSUM_ptr + out_off, s, mask=mask_n)
    tl.store(PSUMSQ_ptr + out_off, sq, mask=mask_n)


@triton.jit
def bn_finalize_apply_kernel(
    X_ptr, OUT_ptr,
    PSUM_ptr, PSUMSQ_ptr,
    WEIGHT_ptr, BIAS_ptr,
    RUN_MEAN_ptr, RUN_VAR_ptr,
    M, N, NUM_TILES, eps, momentum,
    stride_xm, stride_xn,
    BLOCK_N: tl.constexpr, BLOCK_M_APPLY: tl.constexpr, TILE_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Reduce partials
    psum = tl.zeros((BLOCK_N,), dtype=tl.float32)
    psumsq = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for t in range(0, TILE_DIM):
        valid = t < NUM_TILES
        p = tl.load(PSUM_ptr + t * N + offs_n, mask=mask_n & valid, other=0.0)
        ps = tl.load(PSUMSQ_ptr + t * N + offs_n, mask=mask_n & valid, other=0.0)
        psum += p
        psumsq += ps

    mean = psum / M
    var = psumsq / M - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(WEIGHT_ptr + offs_n, mask=mask_n, other=0.0)
    b = tl.load(BIAS_ptr + offs_n, mask=mask_n, other=0.0)
    scale = invstd * w
    shift = b - mean * scale

    # Update running stats only on first row tile
    if pid_m == 0:
        rm = tl.load(RUN_MEAN_ptr + offs_n, mask=mask_n, other=0.0)
        rv = tl.load(RUN_VAR_ptr + offs_n, mask=mask_n, other=0.0)
        # unbiased var for running_var
        unbiased = var * (M / (M - 1))
        new_rm = (1.0 - momentum) * rm + momentum * mean
        new_rv = (1.0 - momentum) * rv + momentum * unbiased
        tl.store(RUN_MEAN_ptr + offs_n, new_rm, mask=mask_n)
        tl.store(RUN_VAR_ptr + offs_n, new_rv, mask=mask_n)

    # Apply
    offs_m = pid_m * BLOCK_M_APPLY + tl.arange(0, BLOCK_M_APPLY)
    mask_m = offs_m < M
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    out_ptrs = OUT_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    y = x * scale[None, :] + shift[None, :]
    tl.store(out_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


def gemm_scaled_eval_bn(x, W, bias_gemm, scale, bn_weight, bn_bias, running_mean, running_var, eps):
    """Compute: y = bn_eval((x @ W^T + bias_gemm) * scale)
    Fully fused affine epilogue.
    """
    M, K = x.shape
    N = W.shape[0]
    # scale and BN are pure affine -> fold
    # acc = x @ W^T   (note: stored as W [N, K])
    # output = ((acc + bias_gemm) * scale - running_mean) * invstd * bn_weight + bn_bias
    # = acc * (scale * invstd * bn_weight) + ((bias_gemm * scale - running_mean) * invstd * bn_weight + bn_bias)
    invstd = 1.0 / torch.sqrt(running_var + eps)
    a_eff = scale * invstd * bn_weight  # [N]
    b_eff = (bias_gemm * scale - running_mean) * invstd * bn_weight + bn_bias  # [N]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # W is [N, K]; we want B as [K, N], stride_bk = 1 (row stride of W transposed view)
    # Treat W as B with shape [K, N] using strides (1, K)
    stride_bk = 1
    stride_bn = K

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_affine_kernel[grid](
        x, W, a_eff, b_eff, out,
        M, N, K,
        x.stride(0), x.stride(1),
        stride_bk, stride_bn,
        out.stride(0), out.stride(1),
    )
    return out


def gemm_only(x, W, bias_gemm, scale):
    """Compute: out = (x @ W^T + bias_gemm) * scale"""
    M, K = x.shape
    N = W.shape[0]
    # a_eff = scale; b_eff = bias_gemm * scale
    a_eff = scale.contiguous()
    b_eff = bias_gemm * scale

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    stride_bk = 1
    stride_bn = K
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_affine_kernel[grid](
        x, W, a_eff, b_eff, out,
        M, N, K,
        x.stride(0), x.stride(1),
        stride_bk, stride_bn,
        out.stride(0), out.stride(1),
    )
    return out


def bn_train_apply(x, bn_weight, bn_bias, running_mean, running_var, eps, momentum):
    M, N = x.shape
    BLOCK_M = 256
    BLOCK_N = 128
    num_row_tiles = triton.cdiv(M, BLOCK_M)
    num_col_tiles = triton.cdiv(N, BLOCK_N)

    psum = torch.empty((num_row_tiles, N), device=x.device, dtype=torch.float32)
    psumsq = torch.empty((num_row_tiles, N), device=x.device, dtype=torch.float32)

    bn_partial_kernel[(num_col_tiles, num_row_tiles)](
        x, psum, psumsq,
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    out = torch.empty_like(x)
    BLOCK_M_APPLY = 128
    num_apply_row_tiles = triton.cdiv(M, BLOCK_M_APPLY)
    # Round up TILE_DIM to power of 2 for tl.arange-style loops? we just iterate Python range
    TILE_DIM = num_row_tiles

    bn_finalize_apply_kernel[(num_col_tiles, num_apply_row_tiles)](
        x, out,
        psum, psumsq,
        bn_weight, bn_bias,
        running_mean, running_var,
        M, N, num_row_tiles, eps, momentum,
        x.stride(0), x.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_M_APPLY=BLOCK_M_APPLY, TILE_DIM=TILE_DIM,
    )
    return out


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
        W = self.gemm.weight  # [N, K]
        bias_gemm = self.gemm.bias  # [N]
        scale = self.scale  # [N] (assuming scale_shape = (out_features,))
        # Ensure scale is [N]
        scale_flat = scale.view(-1)

        if not self.training:
            return gemm_scaled_eval_bn(
                x, W, bias_gemm, scale_flat,
                self.bn.weight, self.bn.bias,
                self.bn.running_mean, self.bn.running_var,
                self.eps,
            )
        else:
            # GEMM with scale fused, then training BN
            y = gemm_only(x, W, bias_gemm, scale_flat)
            out = bn_train_apply(
                y, self.bn.weight, self.bn.bias,
                self.bn.running_mean, self.bn.running_var,
                self.eps, self.momentum,
            )
            # update num_batches_tracked
            if self.bn.num_batches_tracked is not None:
                self.bn.num_batches_tracked.add_(1)
            return out