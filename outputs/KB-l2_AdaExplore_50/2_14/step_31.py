import torch
import torch.nn as nn
import triton
import triton.language as tl


# Key insight: sum_j (x @ W.T)[i,j] = sum_j sum_k x[i,k] * W[j,k]
#            = sum_k x[i,k] * (sum_j W[j,k])
# So we need: out[i] = (sum_k x[i,k] * w_col_sum[k]) * scale
# where w_col_sum[k] = sum_j W[j,k]
#
# BUT: the safety contract forbids precomputing this reduction at init.
# So we compute the full matmul-equivalent at runtime: every multiply must execute.
#
# We implement a fused kernel that does the GEMM but accumulates the N-reduction
# inline using tl.dot. Specifically: per row tile (BLOCK_M rows), we loop over
# K in BLOCK_K chunks; for each K chunk, we load W[:, k_chunk] tiled over N
# (looping over N) and accumulate a partial sum across N, then dot with x.
# Better: load x tile (BLOCK_M, BLOCK_K), and compute
#   acc[m] += sum_n sum_k x[m,k] * W[n,k]   over the chunk
# We can express the inner part as: for each k, w_sum_k = sum_n W[n,k] (loop over n)
# then acc[m] += sum_k x[m,k] * w_sum_k.
#
# We loop n_tile inside, k outside. For each k chunk:
#   w_sum_chunk[BLOCK_K] = 0
#   for n in 0..N step BLOCK_N: w_sum_chunk += sum over n-axis of W[n:n+BLOCK_N, k_chunk]
#   x_chunk[BLOCK_M, BLOCK_K] = load x[:, k_chunk]
#   acc[BLOCK_M] += x_chunk @ w_sum_chunk
# This executes every multiply equivalent: W[n,k] gets multiplied by x[m,k] via
# the distributive form. The "sum" of W down N is a runtime reduction, not an
# init-time precompute.


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_rowsum_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)

    # Compute partial GEMM: P[m, n] = sum_k x[m,k] * W[n,k]   for n in tile
    # Then accumulate sum_n P[m,n] and atomic add to out[m].
    acc_mn = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_row_ptr = x_ptr + offs_m[:, None] * stride_xm
    w_row_ptr = w_ptr + offs_n[:, None] * stride_wn

    for k_start in range(0, K, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < K

        x_ptrs = x_row_ptr + cur_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        w_ptrs = w_row_ptr + cur_k[None, :] * stride_wk
        w_tile = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        # x_tile: (BLOCK_M, BLOCK_K); w_tile: (BLOCK_N, BLOCK_K)
        acc_mn += tl.dot(x_tile, tl.trans(w_tile), out_dtype=tl.float32)

    # Reduce over n dimension and apply scale
    # Mask out invalid n entries
    acc_mn = tl.where(mask_n[None, :], acc_mn, 0.0)
    partial = tl.sum(acc_mn, axis=1) * SCALE

    # Atomic add into out[m]
    tl.atomic_add(out_ptr + offs_m, partial, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = float(scaling_factor)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous()
        w = self.weight
        M, K = x.shape
        N, Kw = w.shape
        assert K == Kw

        scale = self.scaling_factor * 0.5

        out = torch.zeros((M, 1), device=x.device, dtype=x.dtype)
        out_flat = out.view(M)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        fused_gemm_rowsum_kernel[grid](
            x, w, out_flat,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            SCALE=scale,
        )
        return out