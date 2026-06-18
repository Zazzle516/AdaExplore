import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_pool_partial_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    M, N, K, P,  # P = N // KERNEL_SIZE (pooled width)
    num_n_tiles,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pm, stride_pn,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # Mask out-of-range N positions with -inf so they don't affect max
    acc = tl.where(mask_n[None, :], acc, float('-inf'))

    # Reshape for max-pool: (BLOCK_M, BLOCK_N/KS, KS)
    BLOCK_P: tl.constexpr = BLOCK_N // KERNEL_SIZE
    acc_r = tl.reshape(acc, (BLOCK_M, BLOCK_P, KERNEL_SIZE))
    pooled = tl.max(acc_r, axis=2)  # (BLOCK_M, BLOCK_P)

    # Sum along the pooled dimension within this tile
    partial = tl.sum(pooled, axis=1)  # (BLOCK_M,)

    # Store partial sum to partial[m, pid_n]
    p_ptrs = partial_ptr + offs_m * stride_pm + pid_n * stride_pn
    tl.store(p_ptrs, partial, mask=mask_m)


@triton.jit
def reduce_scale_kernel(
    partial_ptr, out_ptr,
    M, NTILES,
    stride_pm, stride_pn,
    scale,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    mask_t = offs_t < NTILES
    p_ptrs = partial_ptr + pid_m * stride_pm + offs_t * stride_pn
    vals = tl.load(p_ptrs, mask=mask_t, other=0.0)
    s = tl.sum(vals, axis=0)
    s = s * scale
    tl.store(out_ptr + pid_m, s)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)

        lin = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(lin.weight.detach().cuda().contiguous())
        self.bias = nn.Parameter(lin.bias.detach().cuda().contiguous())

    def forward(self, x):
        x = x.cuda().contiguous()
        B = x.shape[0]
        K = self.in_features
        N = self.out_features
        KS = self.kernel_size
        P = N // KS

        # Determine BLOCK_N from autotune; we need partial buffer sized to max num_n_tiles.
        # Strategy: launch kernel and allocate partial buffer with conservative N tile count.
        # We'll use a fixed BLOCK_N for partial layout by picking from grid lambda.

        # We allocate partial assuming worst-case BLOCK_N=64 (smallest in configs)
        # then reduce. But autotune chooses BLOCK_N dynamically. To handle this,
        # we compute num_n_tiles inside the grid lambda and size partial accordingly.

        # Simplest: pre-pick BLOCK_N by allocating partial of shape [B, ceil(N/min_BLOCK_N)]
        # but reduce kernel needs exact num_n_tiles. We pass it from meta.

        # Use a 2-step approach: allocate max-sized partial buffer.
        MAX_BLOCK_N = 64  # smallest BLOCK_N -> largest num_n_tiles
        max_tiles = (N + MAX_BLOCK_N - 1) // MAX_BLOCK_N
        partial = torch.empty((B, max_tiles), device=x.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(B, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )

        # We need num_n_tiles to match chosen BLOCK_N. Hack: run kernel and capture.
        # Better: fix BLOCK_N by not using autotune over BLOCK_N — but we already do.
        # Approach: do the launch then read back num_n_tiles via best_config.
        compiled = fused_gemm_pool_partial_kernel[grid](
            x, self.weight, self.bias, partial,
            B, N, K, P,
            0,  # num_n_tiles placeholder, unused in kernel
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            partial.stride(0), partial.stride(1),
            KERNEL_SIZE=KS,
        )

        best = fused_gemm_pool_partial_kernel.best_config
        chosen_BLOCK_N = best.kwargs['BLOCK_N']
        num_n_tiles = (N + chosen_BLOCK_N - 1) // chosen_BLOCK_N

        # Reduce
        out = torch.empty((B,), device=x.device, dtype=torch.float32)
        BLOCK_T = triton.next_power_of_2(num_n_tiles)
        reduce_scale_kernel[(B,)](
            partial, out,
            B, num_n_tiles,
            partial.stride(0), partial.stride(1),
            self.scale_factor,
            BLOCK_T=BLOCK_T,
        )
        return out