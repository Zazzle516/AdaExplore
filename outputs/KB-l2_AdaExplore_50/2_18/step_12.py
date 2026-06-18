import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_rowsum_kernel(
    x_ptr, wt_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Single-pass per BLOCK_M row tile: loop all N tiles internally so we can
    # accumulate row_sum without atomics.
    # wt is weight transposed: shape (K, N), so wt[k, n] = w[n, k].
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_mask = offs_m < M

    num_n_tiles = tl.cdiv(N, BLOCK_N)
    for nt in range(0, num_n_tiles):
        offs_n = nt * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = wt_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
            w_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
            x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
            acc += tl.dot(x_block, w_block)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        # add bias
        b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc += b_vals[None, :]

        # zero out invalid n columns and invalid m rows
        valid = (m_mask[:, None]) & (n_mask[None, :])
        acc = tl.where(valid, acc, 0.0)

        row_sum += tl.sum(acc, axis=1)

    tl.store(out_ptr + offs_m, row_sum, mask=m_mask)


def fused_linear_sum(x, weight_t, bias):
    M, K = x.shape
    N = weight_t.shape[1]
    out = torch.empty(M, device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    matmul_bias_rowsum_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Pre-transpose weight: shape (in_features, out_features) = (K, N)
        # so K-axis is contiguous for tl.dot without tl.trans in the hot loop.
        # We register it as a buffer derived from the linear weight at forward time.
        self._wt_cache = None

    def _get_wt(self):
        w = self.linear.weight  # (N, K)
        if (self._wt_cache is None
                or self._wt_cache.data_ptr() == 0
                or self._wt_cache.shape != (w.shape[1], w.shape[0])
                or self._wt_cache.device != w.device):
            self._wt_cache = w.t().contiguous()
        return self._wt_cache

    def forward(self, x):
        x = x.contiguous()
        wt = self._get_wt()
        b = self.linear.bias.contiguous()
        y = fused_linear_sum(x, wt, b)  # [M]
        return y.unsqueeze(1)