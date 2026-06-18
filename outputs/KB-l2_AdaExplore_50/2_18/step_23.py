import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_rowsum_kernel(
    x_ptr, w_ptr, b_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Real GEMM: computes Y[m,n] = sum_k x[m,k] * w[n,k] + b[n]
    # Then fuses row-sum: out[m] += sum_n Y[m,n]  (via atomic add per (m, N-tile))
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
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_mask = (offs_k[None, :] + k) < K
        x_vals = tl.load(x_ptrs + k * stride_xk, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
        w_vals = tl.load(w_ptrs + k * stride_wk, mask=(offs_n[:, None] < N) & k_mask, other=0.0)
        acc += tl.dot(x_vals, tl.trans(w_vals))

    # add bias (each N-tile owns disjoint n values, so bias[n] is added exactly once per row)
    b_vals = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b_vals[None, :]

    # mask out-of-range N before reducing
    n_mask = offs_n < N
    acc = tl.where(n_mask[None, :], acc, 0.0)

    # sum across BLOCK_N (per-tile partial sum across n)
    row_partial = tl.sum(acc, axis=1)

    m_mask = offs_m < M
    tl.atomic_add(out_ptr + offs_m, row_partial, mask=m_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        W = self.linear.weight  # (N, K)
        B = self.linear.bias    # (N,)

        # Run the linear (GEMM) using cuBLAS (highly tuned for fp32 on RTX 4090).
        y = torch.addmm(B, x, W.t())  # (M, N)

        # Row-sum reduction via Triton (BLOCK_N tuned for N=8192).
        out = torch.empty((M,), device=x.device, dtype=torch.float32)
        BLOCK_N = 1024
        grid = (M,)
        _row_sum_kernel[grid](y, out, M, N, y.stride(0), BLOCK_N=BLOCK_N)

        return out.view(M, 1)


@triton.jit
def _row_sum_kernel(y_ptr, out_ptr, M, N, stride_ym, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_ptr = y_ptr + pid * stride_ym
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        vals = tl.load(row_ptr + offs, mask=mask, other=0.0)
        acc += vals
    s = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid, s)