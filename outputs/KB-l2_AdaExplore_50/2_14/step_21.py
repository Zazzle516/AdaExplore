import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
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
    # Each program handles BLOCK_M rows, iterates over all N tiles
    pid_m = tl.program_id(0)
    pid_split = tl.program_id(1)
    num_splits = tl.num_programs(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Each split handles a portion of the N dimension
    n_per_split = tl.cdiv(N, num_splits)
    n_start = pid_split * n_per_split
    n_end = tl.minimum(n_start + n_per_split, N)

    # Row accumulator (sum over N)
    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # iterate over N tiles within this split
    n_tile = n_start
    num_n_tiles = tl.cdiv(n_end - n_start, BLOCK_N)
    for nt in range(0, num_n_tiles):
        offs_n = n_tile + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_end

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K loop
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k * BLOCK_K + offs_k
            k_mask = k_offs < K

            # Load x: [BLOCK_M, BLOCK_K]
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            x_mask = mask_m[:, None] & k_mask[None, :]
            x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # Load w: weight is (N, K), we want w[offs_n, k_offs] -> [BLOCK_N, BLOCK_K]
            # then we transpose for dot: x @ w.T -> [BLOCK_M, BLOCK_N]
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
            w_mask = mask_n[:, None] & k_mask[None, :]
            w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

            acc += tl.dot(x_block, tl.trans(w_block), allow_tf32=False)

        # mask out invalid n columns and sum
        acc = tl.where(mask_n[None, :], acc, 0.0)
        row_acc += tl.sum(acc, axis=1)

        n_tile += BLOCK_N

    # Apply scale (divide by 2, then multiply by scaling_factor)
    row_acc = row_acc * SCALE

    # atomic add into output [M, 1]
    out_ptrs = out_ptr + offs_m
    tl.atomic_add(out_ptrs, row_acc, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = float(scaling_factor)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        assert x.is_cuda, "input must be CUDA"
        x = x.contiguous()
        w = self.weight.contiguous()
        M, K = x.shape
        N = w.shape[0]
        assert w.shape[1] == K

        out = torch.zeros((M, 1), dtype=torch.float32, device=x.device)

        scale = 0.5 * self.scaling_factor

        # Use a split along N dimension for better parallelism
        # M is small (1024), N is large (8192)
        NUM_SPLITS = 4

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), NUM_SPLITS)

        fused_matmul_sum_kernel[grid](
            x, w, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            SCALE=scale,
        )

        return out