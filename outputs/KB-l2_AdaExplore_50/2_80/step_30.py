import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Fused GEMM + row-max kernel: computes Y = X @ W^T + b, and per output tile,
# reduces along the N dimension on-the-fly to maintain a running row-max.
# To avoid cross-program races we use atomic_max on a global per-row buffer.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_rowmax_kernel(
    x_ptr, w_ptr, b_ptr, rowmax_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)

    x_ptrs = x_ptr + offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_bn[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            x = tl.load(x_ptrs)
            w = tl.load(w_ptrs)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            x_mask = offs_k[None, :] < k_remaining
            w_mask = offs_k[:, None] < k_remaining
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]

    # Mask out-of-bounds N positions to -inf so they don't affect the max
    n_mask = offs_n < N
    neg_inf = float('-inf')
    acc = tl.where(n_mask[None, :], acc, neg_inf)

    # Reduce along N axis within this tile -> (BLOCK_M,)
    tile_max = tl.max(acc, axis=1)

    # Atomic max into per-row buffer (only for valid M rows)
    m_mask = offs_m < M
    tl.atomic_max(rowmax_ptr + offs_m, tile_max, mask=m_mask)


# Apply the post-ops: out[m,0] = gelu(rm[m] - rm[m]) = 0, but we execute it.
@triton.jit
def row_post_kernel(
    rm_ptr, out_ptr,
    M,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    vals = tl.load(rm_ptr + offs, mask=mask, other=0.0)
    x = vals - vals
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    tl.store(out_ptr + offs, gelu, mask=mask)


# Standalone GEMM (fallback path for non max_dim==1)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)

    x_ptrs = x_ptr + offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_bn[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            x = tl.load(x_ptrs)
            w = tl.load(w_ptrs)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            x_mask = offs_k[None, :] < k_remaining
            w_mask = offs_k[:, None] < k_remaining
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def col_max_kernel(
    y_ptr, out_ptr,
    M, N,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= N:
        return
    offs_m = tl.arange(0, BLOCK_M)
    acc = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        idx = m_start + offs_m
        mask = idx < M
        vals = tl.load(y_ptr + idx * stride_ym + pid_n * stride_yn, mask=mask, other=-float('inf'))
        acc = tl.maximum(acc, vals)
    mv = tl.max(acc, axis=0)
    tl.store(out_ptr + pid_n, mv)


@triton.jit
def post_max_dim0_kernel(
    cm_ptr, out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        idx = n_start + offs
        mask = idx < N
        vals = tl.load(cm_ptr + idx, mask=mask, other=0.0)
        acc += vals
    s = tl.sum(acc, axis=0)
    mean = s / N

    for n_start in range(0, N, BLOCK_N):
        idx = n_start + offs
        mask = idx < N
        vals = tl.load(cm_ptr + idx, mask=mask, other=0.0)
        x = vals - mean
        inv_sqrt2 = 0.7071067811865475
        gelu = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
        tl.store(out_ptr + idx, gelu, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.gemm.weight.contiguous()
        b = self.gemm.bias.contiguous()
        M, K = x.shape
        N = w.shape[0]

        if self.max_dim == 1:
            # Fused GEMM + row-max via atomic_max into a per-row buffer
            row_max = torch.full((M,), float('-inf'), device=x.device, dtype=torch.float32)
            even_k = (K % 64 == 0)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
            gemm_rowmax_kernel[grid](
                x, w, b, row_max,
                M, N, K,
                x.stride(0), x.stride(1),
                w.stride(0), w.stride(1),
                EVEN_K=even_k,
            )
            out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
            BLOCK = 256
            grid2 = (triton.cdiv(M, BLOCK),)
            row_post_kernel[grid2](row_max, out, M, BLOCK=BLOCK)
            return out
        elif self.max_dim == 0:
            y = torch.empty((M, N), device=x.device, dtype=x.dtype)
            even_k = (K % 64 == 0)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
            gemm_kernel[grid](
                x, w, b, y,
                M, N, K,
                x.stride(0), x.stride(1),
                w.stride(0), w.stride(1),
                y.stride(0), y.stride(1),
                EVEN_K=even_k,
            )
            col_max = torch.empty((N,), device=x.device, dtype=x.dtype)
            col_max_kernel[(N,)](y, col_max, M, N, y.stride(0), y.stride(1), BLOCK_M=1024)
            out = torch.empty((1, N), device=x.device, dtype=x.dtype)
            post_max_dim0_kernel[(1,)](col_max, out, N, BLOCK_N=1024)
            return out
        else:
            y = self.gemm(x)
            xm = torch.max(y, dim=self.max_dim, keepdim=True).values
            xm = xm - xm.mean(dim=1, keepdim=True)
            return F.gelu(xm)