import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_rowsum_kernel(
    x_ptr, wt_ptr, b_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Real GEMM with row-sum fused into the epilogue.
    # Y[m,n] = sum_k x[m,k] * Wt[k,n] + b[n]
    # out[m] += sum_n Y[m,n]   (atomic add per N-tile)
    # wt_ptr points to W.t() of shape (K, N), contiguous on N.
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

    rm = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)

    x_ptrs = x_ptr + rm[:, None] * stride_xm + offs_k[None, :] * stride_xk
    wt_ptrs = wt_ptr + offs_k[:, None] * stride_wtk + rn[None, :] * stride_wtn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x_vals = tl.load(x_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        w_vals = tl.load(wt_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(x_vals, w_vals)
        x_ptrs += BLOCK_K * stride_xk
        wt_ptrs += BLOCK_K * stride_wtk

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
        self._wt_cache = None
        self._wt_version = None

    def _get_wt(self):
        W = self.linear.weight  # (N, K)
        v = W._version
        if self._wt_cache is None or self._wt_version != v or not self._wt_cache.is_cuda:
            self._wt_cache = W.detach().t().contiguous().cuda()
            self._wt_version = v
        return self._wt_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        Wt = self._get_wt()  # (K, N) contiguous
        B = self.linear.bias.contiguous().cuda()

        out = torch.zeros((M,), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        )
        fused_gemm_rowsum_kernel[grid](
            x, Wt, B, out,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
        )

        # After sum: shape (M, 1). Max/mean/logsumexp over size-1 dim are identity.
        return out.view(M, 1)