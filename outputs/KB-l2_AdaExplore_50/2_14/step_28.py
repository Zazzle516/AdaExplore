import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Compute y = x @ W^T  (M, N), then sum over N axis, scale.
    # Each program handles a tile of BLOCK_M rows.
    # Loops over N in BLOCK_N chunks; for each chunk runs the full K reduction
    # via tl.dot, accumulating sum along N into a per-row accumulator.

    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        gemm_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            k_mask = k_offs < K
            # x tile (BLOCK_M, BLOCK_K)
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
            # w tile (BLOCK_N, BLOCK_K) -> need (BLOCK_K, BLOCK_N) for dot
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
            w_tile = tl.load(w_ptrs, mask=mask_n[:, None] & k_mask[None, :], other=0.0)
            gemm_acc += tl.dot(x_tile, tl.trans(w_tile))

        # Sum along N axis (mask out-of-range columns)
        gemm_acc = tl.where(mask_n[None, :], gemm_acc, 0.0)
        row_acc += tl.sum(gemm_acc, axis=1)

    row_acc = row_acc * SCALE
    tl.store(out_ptr + offs_m, row_acc, mask=mask_m)


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

        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
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