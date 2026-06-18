import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 512}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program computes a tile of rows of x, and sums over all N (hidden)
    # Result per row: sum_n( sum_k x[m,k] * w[n,k] ) * scale
    # = sum_k x[m,k] * (sum_n w[n,k]) * scale
    # But we must NOT precompute sum_n w[n,k]. So we do it at runtime here:
    # we still iterate the full GEMM but reduce on the fly.
    #
    # Strategy: for this tile of M rows, loop over K in BLOCK_K chunks.
    # For each K chunk, compute w_col_sum[k] = sum_n w[n, k] inside kernel.
    # Then accumulate x[m, k] * w_col_sum[k] into row accumulator.
    # This keeps all ops at runtime (no init-time precompute).

    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    BLOCK_N: tl.constexpr = 128

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Compute w_col_sum for this k-chunk by iterating over N
        w_col_sum = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for n_start in range(0, N, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            # w shape (N, K): w[n, k]
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w_mask = mask_n[:, None] & mask_k[None, :]
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
            w_col_sum += tl.sum(w_tile, axis=0)

        # Load x tile (BLOCK_M, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Multiply and reduce over K
        partial = x_tile * w_col_sum[None, :]
        acc += tl.sum(partial, axis=1)

    acc = acc * SCALE
    tl.store(out_ptr + offs_m, acc, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = float(scaling_factor)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous()
        w = self.weight.contiguous()
        M, K = x.shape
        N = w.shape[0]
        # scale = scaling_factor / 2 (from the /2 division)
        scale = self.scaling_factor * 0.5

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        out_flat = out.view(M)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        fused_kernel[grid](
            x, w, out_flat,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            SCALE=scale,
        )
        return out