import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
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
    # Grid: (M/BLOCK_M, N/BLOCK_N). Each program computes a (BLOCK_M, BLOCK_N) tile
    # of the matmul Y = X @ W^T, then reduces across the N dim (BLOCK_N axis) and
    # atomic-adds into a per-row accumulator. Final scaling is applied after.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_remaining = K - k_start
        x_mask = mask_m[:, None] & (offs_k[None, :] < k_remaining)
        w_mask = mask_n[:, None] & (offs_k[None, :] < k_remaining)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        # x_tile: (BLOCK_M, BLOCK_K), w_tile: (BLOCK_N, BLOCK_K) -> need (BLOCK_K, BLOCK_N)
        acc += tl.dot(x_tile, tl.trans(w_tile))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # mask out invalid N columns before reduction
    acc = tl.where(mask_n[None, :], acc, 0.0)
    row_sum = tl.sum(acc, axis=1) * SCALE  # (BLOCK_M,)

    # atomic add into per-row output
    tl.atomic_add(out_ptr + offs_m, row_sum, mask=mask_m)


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
        scale = self.scaling_factor * 0.5

        out = torch.zeros((M, 1), device=x.device, dtype=torch.float32)
        out_flat = out.view(M)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        fused_kernel[grid](
            x, w, out_flat,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            SCALE=scale,
        )
        return out