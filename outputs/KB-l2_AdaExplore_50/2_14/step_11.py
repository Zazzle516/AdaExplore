import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    scale,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Compute: out[m] = scale * sum_n( (x[m,:] @ w[n,:]) ) / 2
    # = (scale/2) * sum_k( x[m,k] * sum_n(w[n,k]) )
    # But we must execute the full matmul. So accumulate matmul then reduce.
    # Strategy: each program handles BLOCK_M rows. Loop over N in chunks of BLOCK_N,
    # accumulating per-row sum of (x @ w^T) values.
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # We accumulate sum over N of (x[m] dot w[n]) which equals x[m] dot (sum_n w[n])
    # To respect "every operator executes", we still do per-n dot products and sum.
    # Loop over N one block at a time.
    BLOCK_N: tl.constexpr = 64

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        # Compute partial dots for this n block over all K
        # partial[m,n] = sum_k x[m,k] * w[n,k]
        partial = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_vals = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w_vals = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            # x: [BM, BK], w: [BN, BK] -> we want [BM, BN]
            partial += tl.dot(x_vals, tl.trans(w_vals))
        # divide by 2 and sum over n
        partial = partial * 0.5
        partial = tl.where(mask_n[None, :], partial, 0.0)
        acc += tl.sum(partial, axis=1)

    acc = acc * scale
    tl.store(out_ptr + offs_m, acc, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = float(scaling_factor)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.weight.contiguous()
        M, K = x.shape
        N = w.shape[0]
        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        fused_kernel[grid](
            x, w, out,
            M, N, K,
            self.scaling_factor,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
        )
        return out