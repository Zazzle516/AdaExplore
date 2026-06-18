import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # one program per row tile; reduces over N inside
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rm_mask = rm < M

    # accumulator: per-row sum of (x @ W^T) elements
    row_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over K in chunks; for each chunk, compute partial dot with summed-weights would be a shortcut.
    # Instead, we loop over N (output cols) tile, compute full GEMM tile, then sum into row_acc.
    # But we need to also iterate over K. Use a 2-level approach: outer over N blocks, inner over K.
    # To keep one kernel call efficient: tile over (M, N) with K reduction, accumulate sum over N.

    # We'll iterate N in BLOCK_N chunks
    BLOCK_N: tl.constexpr = 32

    for n_start in range(0, N, BLOCK_N):
        rn = n_start + tl.arange(0, BLOCK_N)
        rn_mask = rn < N

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            k_mask = rk < K

            x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
            x_mask = rm_mask[:, None] & k_mask[None, :]
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)

            w_ptrs = w_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk
            w_mask = rn_mask[:, None] & k_mask[None, :]
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)

            acc += tl.dot(x, tl.trans(w))

        # divide by 2 and sum over N tile
        acc = acc * 0.5
        # mask out invalid n
        acc = tl.where(rn_mask[None, :], acc, 0.0)
        row_acc += tl.sum(acc, axis=1)

    # scale
    row_acc = row_acc * scale
    out_ptrs = out_ptr + rm
    tl.store(out_ptrs, row_acc, mask=rm_mask)


def fused_matmul_div_sum_scale(x: torch.Tensor, w: torch.Tensor, scaling_factor: float):
    M, K = x.shape
    N, K2 = w.shape
    assert K == K2
    x = x.contiguous()
    w = w.contiguous()
    out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    fused_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        scaling_factor,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        return fused_matmul_div_sum_scale(x, self.weight, self.scaling_factor)