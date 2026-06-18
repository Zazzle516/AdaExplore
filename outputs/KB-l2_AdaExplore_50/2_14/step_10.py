import torch
import torch.nn as nn
import triton
import triton.language as tl


# The key insight: output is (batch, 1) where
# out[b] = scaling_factor * sum_j (x[b,:] @ W[j,:]) / 2
#       = (scaling_factor/2) * sum_j sum_k x[b,k] * W[j,k]
#       = (scaling_factor/2) * sum_k x[b,k] * (sum_j W[j,k])
#
# BUT per the safety contract, we cannot precompute sum_j W[j,k] at init time.
# We must execute the matmul at runtime. So we do the full matmul fused with
# the row-sum reduction and scaling.
#
# Strategy: one program per batch row. The program iterates over K in tiles,
# loads x[b, k_tile] and W[:, k_tile] (all hidden rows for that k tile),
# computes the partial matmul-then-sum reduction. Specifically for each k:
#   contrib_k = x[b,k] * sum_j W[j,k]
# We compute this by tiling over K and J: for each k-tile we need sum over
# all j of W[j, k_tile]. We do this with a tl.dot of a ones vector against W.
# Actually simpler: out[b] = sum_k x[b,k] * col_sum[k], where col_sum is
# computed at runtime by summing W along dim 0.
#
# To comply with safety: compute col_sum at runtime each forward (not at init).
# But that still rewrites the graph. Better: do the actual matmul producing
# intermediate (B, H), then sum it. We'll use a fused kernel that for each
# (batch, k-tile) loads W tile and reduces internally.

# Approach: tiled GEMM fused with the row-sum reduction.
# For each batch row b, one program computes:
#   acc = 0
#   for k_tile in K:
#     x_tile = x[b, k_tile]            # [BK]
#     for j_tile in H:
#       w_tile = W[j_tile, k_tile]     # [BJ, BK]
#       # partial result: y_partial[j_tile] += sum_k w_tile * x_tile -> [BJ]
#       # then sum over j_tile, add to acc
#   out[b] = acc * scale
#
# We can fuse: acc += sum over j_tile, k_tile of W[j,k] * x[k]
#            = sum_k x[k] * (sum_j W[j,k]) over the tile
# So per program we just accumulate scalar. Use tl.dot to be efficient:
#   compute ones[BJ] @ W[BJ, BK] = colsum_tile[BK], then dot with x_tile.
# Or equivalently dot W[BJ, BK] with x[BK] -> [BJ], then sum.

@triton.jit
def fused_kernel(
    x_ptr, w_ptr, out_ptr,
    B, H, K,
    scale,
    stride_xb, stride_xk,
    stride_wh, stride_wk,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)  # batch tile index

    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_k = tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)

    b_mask = offs_b < B

    row_acc = tl.zeros((BLOCK_B,), dtype=tl.float32)

    for h_start in range(0, H, BLOCK_H):
        h_idx = h_start + offs_h
        h_mask = h_idx < H

        for k_start in range(0, K, BLOCK_K):
            k_idx = k_start + offs_k
            k_mask = k_idx < K

            # Load x tile: [BLOCK_B, BLOCK_K]
            x_ptrs = x_ptr + offs_b[:, None] * stride_xb + k_idx[None, :] * stride_xk
            x_mask = b_mask[:, None] & k_mask[None, :]
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # Load W tile: [BLOCK_H, BLOCK_K]
            w_ptrs = w_ptr + h_idx[:, None] * stride_wh + k_idx[None, :] * stride_wk
            w_mask = h_mask[:, None] & k_mask[None, :]
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

            # x_tile @ w_tile.T -> [BLOCK_B, BLOCK_H]
            partial = tl.dot(x_tile, tl.trans(w_tile), out_dtype=tl.float32, allow_tf32=False)
            # Sum over H axis immediately, accumulate scalar per batch row
            row_acc += tl.sum(partial, axis=1)

    out_val = row_acc * scale
    tl.store(out_ptr + offs_b, out_val, mask=b_mask)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.weight.contiguous().cuda()
        B, K = x.shape
        H, Kw = w.shape
        assert K == Kw

        out = torch.empty((B, 1), device=x.device, dtype=x.dtype)

        scale = self.scaling_factor * 0.5

        BLOCK_B = 32
        BLOCK_K = 64
        BLOCK_H = 128

        grid = ((B + BLOCK_B - 1) // BLOCK_B,)
        fused_kernel[grid](
            x, w, out,
            B, H, K,
            scale,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            BLOCK_B=BLOCK_B,
            BLOCK_H=BLOCK_H,
            BLOCK_K=BLOCK_K,
            num_warps=8,
            num_stages=3,
        )
        return out