import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_rowsum_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # one program per M tile; reduces over N tiles internally
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    num_n = tl.cdiv(N, BLOCK_N)
    num_k = tl.cdiv(K, BLOCK_K)

    for n_idx in range(0, num_n):
        n_start = n_idx * BLOCK_N
        cur_n = n_start + offs_n
        n_mask = cur_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_idx in range(0, num_k):
            k_start = k_idx * BLOCK_K
            cur_k = k_start + offs_k
            k_mask = cur_k < K

            # x: (M, K) -> tile (BLOCK_M, BLOCK_K)
            x_ptrs = x_ptr + offs_m[:, None] * K + cur_k[None, :]
            x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            # w: (N, K) -> tile (BLOCK_N, BLOCK_K); we need W.T so use (K, N) layout via transpose
            w_ptrs = w_ptr + cur_n[:, None] * K + cur_k[None, :]
            w_tile = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

            acc += tl.dot(x_tile, tl.trans(w_tile))

        # reduce over N within this tile, accumulate into row_acc
        row_acc += tl.sum(acc, axis=1)

    row_acc = row_acc * SCALE
    tl.store(out_ptr + offs_m, row_acc, mask=m_mask)


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

        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
        out_flat = out.view(M)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        fused_matmul_rowsum_kernel[grid](
            x, w, out_flat,
            M, N, K,
            SCALE=scale,
        )
        return out