import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_linear_rowsum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per BLOCK_M rows. Loops over N tiles internally and
    # accumulates a per-row sum entirely in registers (no atomics).
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n0 = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M

    # Preload x rows into registers? Too big (K=8192). Instead loop tiles of N,
    # and within each N tile do the K reduction with tl.dot.
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    # bias sum over all valid n (handled per-tile below)

    x_row_ptrs = x_ptr + offs_m[:, None] * stride_xm  # (BLOCK_M, 1)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + offs_n0
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        w_base = w_ptr + offs_n[None, :] * stride_wn  # (1, BLOCK_N)

        for k in range(0, K, BLOCK_K):
            k_idx = k + offs_k
            x_vals = tl.load(x_row_ptrs + k_idx[None, :] * stride_xk,
                             mask=m_mask[:, None], other=0.0)
            w_vals = tl.load(w_base + k_idx[:, None] * stride_wk,
                             mask=n_mask[None, :], other=0.0)
            acc += tl.dot(x_vals, w_vals, allow_tf32=True)

        b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc += b_vals[None, :]
        acc = tl.where(n_mask[None, :], acc, 0.0)
        row_sum += tl.sum(acc, axis=1)

    tl.store(out_ptr + offs_m, row_sum, mask=m_mask)


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

        W = self.linear.weight.contiguous()  # (N, K)
        B = self.linear.bias.contiguous()    # (N,)

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)

        _fused_linear_rowsum_kernel[grid](
            x, W, B, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
        )

        # After row-sum: out is (M,). Subsequent ops:
        # max over dim=1 keepdim -> (M,1) (same value since size-1)
        # mean over dim=1 -> same
        # logsumexp over dim=1 (size 1) -> same value
        # logsumexp again -> same.
        # So final output is just out reshaped.
        return out.view(M, 1)