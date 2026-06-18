import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_rowsum_kernel(
    x_ptr, wt_ptr, out_ptr,
    M, N, K,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # one program per M tile; reduces over N tiles internally
    # wt_ptr points to W.T which is (K, N) row-major contiguous
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

            # x: (M, K) tile (BLOCK_M, BLOCK_K)
            x_ptrs = x_ptr + offs_m[:, None] * K + cur_k[None, :]
            x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            # wt: (K, N) tile (BLOCK_K, BLOCK_N) - inner dim N contiguous
            wt_ptrs = wt_ptr + cur_k[:, None] * N + cur_n[None, :]
            wt_tile = tl.load(wt_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

            acc += tl.dot(x_tile, wt_tile, allow_tf32=True)

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
        self._wt_cache = None
        self._wt_version = None

    def _get_wt(self):
        # cache the transposed contiguous weight; rebuild if weight changes
        v = self.weight._version
        if self._wt_cache is None or self._wt_version != v or self._wt_cache.device != self.weight.device:
            self._wt_cache = self.weight.detach().t().contiguous()
            self._wt_version = v
        return self._wt_cache

    def forward(self, x):
        x = x.contiguous()
        wt = self._get_wt()  # (K, N) contiguous
        M, K = x.shape
        N = wt.shape[1]
        scale = self.scaling_factor * 0.5

        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
        out_flat = out.view(M)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        fused_matmul_rowsum_kernel[grid](
            x, wt, out_flat,
            M, N, K,
            SCALE=scale,
        )
        return out