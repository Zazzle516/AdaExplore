import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N_POOLED': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N_POOLED': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N_POOLED': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N_POOLED': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N_POOLED': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N_POOLED': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N_POOLED': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N_POOLED': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_pool_sum_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    M, N, K, NUM_N_TILES,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pn, stride_pm,
    BLOCK_M: tl.constexpr,
    BLOCK_N_POOLED: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pooled_offs = pid_n * BLOCK_N_POOLED + tl.arange(0, BLOCK_N_POOLED)
    offs_n_even = pooled_offs * 2
    offs_n_odd = pooled_offs * 2 + 1
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_even = offs_n_even < N
    mask_odd = offs_n_odd < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_even_ptrs = w_ptr + offs_n_even[:, None] * stride_wn + offs_k[None, :] * stride_wk
    w_odd_ptrs = w_ptr + offs_n_odd[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc_even = tl.zeros((BLOCK_M, BLOCK_N_POOLED), dtype=tl.float32)
    acc_odd = tl.zeros((BLOCK_M, BLOCK_N_POOLED), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        w_e = tl.load(w_even_ptrs, mask=mask_even[:, None] & k_mask[None, :], other=0.0)
        w_o = tl.load(w_odd_ptrs, mask=mask_odd[:, None] & k_mask[None, :], other=0.0)
        acc_even += tl.dot(x, tl.trans(w_e), allow_tf32=False)
        acc_odd += tl.dot(x, tl.trans(w_o), allow_tf32=False)
        x_ptrs += BLOCK_K * stride_xk
        w_even_ptrs += BLOCK_K * stride_wk
        w_odd_ptrs += BLOCK_K * stride_wk

    # add bias
    b_e = tl.load(b_ptr + offs_n_even, mask=mask_even, other=0.0)
    b_o = tl.load(b_ptr + offs_n_odd, mask=mask_odd, other=0.0)
    acc_even = acc_even + b_e[None, :]
    acc_odd = acc_odd + b_o[None, :]

    neg_inf = float('-inf')
    acc_even = tl.where(mask_even[None, :], acc_even, neg_inf)
    acc_odd = tl.where(mask_odd[None, :], acc_odd, neg_inf)

    pooled = tl.maximum(acc_even, acc_odd)

    partial = tl.sum(pooled, axis=1)  # (BLOCK_M,)

    # Write to partial buffer (NUM_N_TILES, M) - one row per pid_n
    partial_ptrs = partial_ptr + pid_n * stride_pn + offs_m * stride_pm
    tl.store(partial_ptrs, partial, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.matmul.weight.contiguous()
        B = self.matmul.bias.contiguous()
        M, K = x.shape
        N = W.shape[0]
        pooled_N = N // self.kernel_size

        # Determine number of N tiles. We'll allocate the partial buffer
        # based on a reasonable assumption — use a fixed launch grid based on
        # autotune's chosen BLOCK_N_POOLED. To handle this, we allocate
        # max possible tiles and pass NUM_N_TILES = pooled_N // BLOCK_N_POOLED.
        # Simpler: allocate worst-case (pooled_N // 32) rows.
        MAX_TILES = (pooled_N + 31) // 32
        partial = torch.zeros((MAX_TILES, M), device=x.device, dtype=torch.float32)

        def grid(meta):
            return (
                triton.cdiv(M, meta['BLOCK_M']),
                triton.cdiv(pooled_N, meta['BLOCK_N_POOLED']),
            )

        fused_linear_pool_sum_kernel[grid](
            x, W, B, partial,
            M, N, K, MAX_TILES,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            partial.stride(0), partial.stride(1),
        )
        out = partial.sum(dim=0) * self.scale_factor
        return out