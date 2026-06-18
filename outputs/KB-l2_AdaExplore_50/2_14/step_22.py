import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_sum_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles BLOCK_M rows; iterates over all N tiles, accumulating row-sum.
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # Compute tile [BLOCK_M, BLOCK_N] = x[m, :] @ W[n, :]^T
        tile_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            mask_k = k_offs < K
            # x: (BLOCK_M, BLOCK_K)
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            x_vals = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            # W: (BLOCK_N, BLOCK_K) -> need W^T tile (BLOCK_K, BLOCK_N)
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
            w_vals = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            tile_acc += tl.dot(x_vals, tl.trans(w_vals))

        # Sum tile along N and accumulate into row_acc
        # mask out-of-range N columns
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
        W = self.weight.contiguous()
        M, K = x.shape
        N = W.shape[0]
        assert K == W.shape[1]

        eff_scale = self.scaling_factor * 0.5

        out = torch.empty(M, device=x.device, dtype=torch.float32)

        grid = lambda meta: ((M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)
        fused_matmul_sum_kernel[grid](
            x, W, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            SCALE=eff_scale,
        )
        return out.view(M, 1)