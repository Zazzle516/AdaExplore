import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64, 'BLOCK_K': 32, 'BLOCK_H': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_B': 128, 'BLOCK_K': 64, 'BLOCK_H': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_B': 32, 'BLOCK_K': 64, 'BLOCK_H': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_B': 64, 'BLOCK_K': 64, 'BLOCK_H': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_B': 32, 'BLOCK_K': 128, 'BLOCK_H': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_B': 16, 'BLOCK_K': 64, 'BLOCK_H': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_B': 64, 'BLOCK_K': 32, 'BLOCK_H': 64}, num_warps=4, num_stages=4),
    ],
    key=['B', 'K', 'H'],
)
@triton.jit
def fused_matmul_sum_kernel(
    x_ptr, w_ptr, out_ptr,
    B, K, H,
    scale,
    stride_xb, stride_xk,
    stride_wh, stride_wk,
    BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    b_start = pid * BLOCK_B

    offs_b = b_start + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    # accumulator: per-row sum
    row_acc = tl.zeros((BLOCK_B,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for h_start in range(0, H, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        tile_acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_idx = k_start + offs_k
            mask_k = k_idx < K

            x_ptrs = x_ptr + offs_b[:, None] * stride_xb + k_idx[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=mask_b[:, None] & mask_k[None, :], other=0.0)

            w_ptrs = w_ptr + offs_h[:, None] * stride_wh + k_idx[None, :] * stride_wk
            w_tile = tl.load(w_ptrs, mask=mask_h[:, None] & mask_k[None, :], other=0.0)

            tile_acc += tl.dot(x_tile, tl.trans(w_tile))

        # mask H dim to zero out-of-range cols
        tile_acc = tl.where(mask_h[None, :], tile_acc, 0.0)
        row_acc += tl.sum(tile_acc, axis=1)

    row_acc = row_acc * scale

    out_ptrs = out_ptr + offs_b
    tl.store(out_ptrs, row_acc, mask=mask_b)


def fused_matmul_div_sum_scale(x, w, scaling_factor):
    B, K = x.shape
    H, K2 = w.shape
    assert K == K2

    x = x.contiguous()
    w = w.contiguous()

    out = torch.empty((B,), device=x.device, dtype=torch.float32)
    scale = scaling_factor * 0.5

    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)
    fused_matmul_sum_kernel[grid](
        x, w, out,
        B, K, H,
        scale,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
    )
    return out.view(B, 1)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return fused_matmul_div_sum_scale(x, self.weight, self.scaling_factor)