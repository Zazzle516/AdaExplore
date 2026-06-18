import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_sum_kernel(
    x_ptr, ws_ptr, out_ptr,
    M, N, K,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute: out[m] = SCALE * sum_n sum_k x[m,k] * W[n,k]
                    = SCALE * sum_k x[m,k] * ws[k]
    where ws[k] = sum_n W[n,k] is precomputed.
    
    BUT we must NOT collapse the GEMM at init time (safety contract).
    So ws is computed every forward via a separate kernel.
    
    Actually, the safety contract says don't precompute at INIT. Computing
    a column-sum at runtime each forward is also forbidden if it replaces
    the full op. Let's do real GEMM instead.
    """
    pass


# Real tiled GEMM with fused divide + row-sum + scale epilogue.
# For each (M_tile) program, loop over N in BLOCK_N chunks; for each chunk,
# accumulate full K via tl.dot to get (BLOCK_M, BLOCK_N) tile, then reduce
# along N into a per-row accumulator. After looping over all N, multiply by
# scale and store.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_rowsum_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        tile_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            mask_k = k_offs < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            a = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            w_ptrs = w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
            b = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            tile_acc += tl.dot(a, b)

        # mask out-of-range N
        tile_acc = tl.where(mask_n[None, :], tile_acc, 0.0)
        row_acc += tl.sum(tile_acc, axis=1)

    out_vals = row_acc * SCALE
    tl.store(out_ptr + offs_m, out_vals, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = float(scaling_factor)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous()
        # W: (hidden_size, input_size) -> need W.T which is (input_size, hidden_size)
        # We want B of shape (K, N) = (input_size, hidden_size) for tl.dot(A(M,K), B(K,N))
        # W.T = weight.t() gives that view. To make K-loads on B contiguous along K,
        # we want B with stride_wk = 1 (along K). That's weight stored as (N, K) and
        # accessed transposed: B[k, n] = weight[n, k], so stride_wk = 1, stride_wn = K.
        w = self.weight  # (N, K) = (hidden_size, input_size), contiguous
        M, K = x.shape
        N = w.shape[0]

        scale = self.scaling_factor * 0.5

        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
        out_flat = out.view(M)

        # B[k, n] = w[n, k] -> stride_wk = 1, stride_wn = K = w.stride(0)
        stride_wk = 1
        stride_wn = w.stride(0)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        gemm_rowsum_kernel[grid](
            x, w, out_flat,
            M, N, K,
            x.stride(0), x.stride(1),
            stride_wk, stride_wn,
            SCALE=scale,
        )
        return out