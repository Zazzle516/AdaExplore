import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_logsumexp_fused_kernel(
    x_ptr, wt_ptr, b_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    row_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    num_n_tiles = tl.cdiv(N, BLOCK_N)

    for n_idx in range(0, num_n_tiles):
        offs_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            wt_ptrs = wt_ptr + k_offs[:, None] * stride_wtk + offs_n[None, :] * stride_wtn
            x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            wt_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w = tl.load(wt_ptrs, mask=wt_mask, other=0.0)
            acc += tl.dot(x, w)
        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + b[None, :]
        n_mask = offs_n[None, :] < N
        acc = tl.where(n_mask, acc, -float('inf'))
        tile_max = tl.max(acc, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        # rescale running sum
        scale = tl.exp(row_max - new_max)
        # zero where row_max was -inf will produce nan*0; guard
        scale = tl.where(new_max == float('-inf'), 0.0, scale)
        e = tl.exp(acc - new_max[:, None])
        e = tl.where(n_mask, e, 0.0)
        tile_sum = tl.sum(e, axis=1)
        row_sum = row_sum * scale + tile_sum
        row_max = new_max

    # finalize: logsumexp = row_max + log(row_sum)
    x = row_max + tl.log(row_sum)
    # 2x LeakyReLU(0.01) => slope 0.0001 for negatives
    x = tl.where(x >= 0, x, x * 0.0001)
    # GELU twice (exact via erf)
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    m_mask = offs_m < M
    tl.store(out_ptr + offs_m, x, mask=m_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self._wt_cache = None
        self._wt_version = -1

    def _get_wt(self):
        W = self.linear.weight
        if (self._wt_cache is None) or (self._wt_version != W._version) or (not self._wt_cache.is_cuda):
            wt = W.detach().cuda().t().contiguous()
            self._wt_cache = wt
            self._wt_version = W._version
        return self._wt_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        Wt = self._get_wt()  # (K, N) contiguous
        if self.linear.bias is not None:
            b = self.linear.bias.detach().contiguous().cuda()
        else:
            b = torch.zeros(self.out_features, device=x.device, dtype=x.dtype)

        M, K = x.shape
        N = Wt.shape[1]

        out = torch.empty(M, 1, device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        gemm_logsumexp_fused_kernel[grid](
            x, Wt, b,
            out,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
        )
        return out