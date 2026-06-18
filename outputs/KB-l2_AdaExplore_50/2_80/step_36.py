import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# GEMM kernel: computes Y = X @ W^T + b, where X is (M, K), W is (N, K), b is (N,)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
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

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < k_remaining)
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


# Row-wise max reduction kernel: input (M, N), output (M, 1)
@triton.jit
def row_max_kernel(
    y_ptr, out_ptr,
    M, N,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.full((BLOCK_N,), -float('inf'), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        idx = n_start + offs_n
        mask = idx < N
        vals = tl.load(y_ptr + pid_m * stride_ym + idx * stride_yn, mask=mask, other=-float('inf'))
        acc = tl.maximum(acc, vals)
    m = tl.max(acc, axis=0)
    tl.store(out_ptr + pid_m, m)


# Column-wise max reduction kernel: input (M, N), output (N,)
# Each program handles one column.
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


# For max_dim=1: input shape (M, N) -> max -> (M, 1) -> x - x.mean(dim=1, keepdim=True)
# mean over dim=1 of (M,1) gives same value, so x - mean(x) over the singleton = 0
# Then GELU(0) = 0. Result is (M, 1) of zeros.

# For max_dim=0: input shape (M, N) -> max -> (1, N) -> x - x.mean(dim=1, keepdim=True)
# mean over dim=1 of (1, N) is scalar = mean of column maxes
# Then GELU applied.
@triton.jit
def post_max_dim0_kernel(
    cm_ptr, out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    # Compute mean of cm
    offs = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        idx = n_start + offs
        mask = idx < N
        vals = tl.load(cm_ptr + idx, mask=mask, other=0.0)
        acc += vals
    s = tl.sum(acc, axis=0)
    mean = s / N

    # Now compute output: gelu(cm - mean)
    for n_start in range(0, N, BLOCK_N):
        idx = n_start + offs
        mask = idx < N
        vals = tl.load(cm_ptr + idx, mask=mask, other=0.0)
        x = vals - mean
        # GELU using erf form: 0.5 * x * (1 + erf(x / sqrt(2)))
        inv_sqrt2 = 0.7071067811865475
        gelu = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
        tl.store(out_ptr + idx, gelu, mask=mask)


def triton_gemm(x, w, b):
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_kernel[grid](
        x, w, b, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.in_features = in_features
        self.out_features = out_features
        self._wt_cache = None
        self._wt_version = -1

    def _get_wt(self):
        w = self.gemm.weight
        if self._wt_cache is None or self._wt_version != w._version or self._wt_cache.device != w.device:
            wt = w.detach().t().contiguous().cuda()
            self._wt_cache = wt
            self._wt_version = w._version
        return self._wt_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        wt = self._get_wt()  # (K, N) contiguous
        b = self.gemm.bias.contiguous().cuda()

        # GEMM: y = x @ wt + b. wt has stride (N, 1) -> stride_wn=1, stride_wk=N
        # Reuse gemm_kernel by passing wt with swapped stride args:
        # original kernel expects W as (N,K) with stride_wn, stride_wk
        # but we pass wt as (K,N) and tell kernel stride_wn=1, stride_wk=N
        M, K = x.shape
        N = wt.shape[1]
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        gemm_kernel[grid](
            x, wt, b, y,
            M, N, K,
            x.stride(0), x.stride(1),
            1, wt.stride(0),  # stride_wn=1 (along N contiguous), stride_wk=N
            y.stride(0), y.stride(1),
        )
        M, N = y.shape

        if self.max_dim == 1:
            # max over dim=1 keepdim=True -> (M, 1)
            row_max = torch.empty((M,), device=x.device, dtype=x.dtype)
            grid = (M,)
            row_max_kernel[grid](y, row_max, M, N, y.stride(0), y.stride(1), BLOCK_N=2048)
            xm = row_max.view(M, 1)
            xm = xm - xm.mean(dim=1, keepdim=True)
            return F.gelu(xm)
        elif self.max_dim == 0:
            # max over dim=0 keepdim=True -> (1, N)
            col_max = torch.empty((N,), device=x.device, dtype=x.dtype)
            grid = (N,)
            col_max_kernel[grid](y, col_max, M, N, y.stride(0), y.stride(1), BLOCK_M=1024)
            out = torch.empty((1, N), device=x.device, dtype=x.dtype)
            post_max_dim0_kernel[(1,)](col_max, out, N, BLOCK_N=1024)
            return out
        else:
            # Fallback to PyTorch
            xm = torch.max(y, dim=self.max_dim, keepdim=True).values
            xm = xm - xm.mean(dim=1, keepdim=True)
            return F.gelu(xm)