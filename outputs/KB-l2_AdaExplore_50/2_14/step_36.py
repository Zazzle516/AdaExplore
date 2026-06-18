import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128,'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def partial_kernel(
    x_ptr, w_ptr, partial_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pm, stride_pn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (M/BLOCK_M, N/BLOCK_N). Each program computes a (BLOCK_M, BLOCK_N) tile
    # of Y = X @ W^T, reduces along the BLOCK_N dim and writes (BLOCK_M,) partial
    # to partial[pid_m*BLOCK_M:..., pid_n].
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_remaining = K - k_start
        x_mask = mask_m[:, None] & (offs_k[None, :] < k_remaining)
        w_mask = mask_n[:, None] & (offs_k[None, :] < k_remaining)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x_tile, tl.trans(w_tile))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    acc = tl.where(mask_n[None, :], acc, 0.0)
    row_sum = tl.sum(acc, axis=1)  # (BLOCK_M,)

    # partial shape: (M, num_n_tiles)
    p_ptrs = partial_ptr + offs_m * stride_pm + pid_n * stride_pn
    tl.store(p_ptrs, row_sum, mask=mask_m)


@triton.jit
def reduce_kernel(
    partial_ptr, out_ptr,
    M, NUM_N,
    stride_pm, stride_pn,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n_start in range(0, NUM_N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < NUM_N
        p_ptrs = partial_ptr + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
        p_mask = mask_m[:, None] & mask_n[None, :]
        p_tile = tl.load(p_ptrs, mask=p_mask, other=0.0)
        acc += tl.sum(p_tile, axis=1)

    acc = acc * SCALE
    tl.store(out_ptr + offs_m, acc, mask=mask_m)


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

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        out_flat = out.view(M)

        # Allocate scratch for partial sums; size depends on BLOCK_N selected.
        # We allocate worst case based on smallest BLOCK_N in configs (64).
        max_num_n = triton.cdiv(N, 64)
        partial = torch.empty((M, max_num_n), device=x.device, dtype=torch.float32)

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        partial_kernel[grid](
            x, w, partial,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            partial.stride(0), partial.stride(1),
        )

        # Determine actual num_n used by chosen config
        best_cfg = partial_kernel.best_config
        block_n = best_cfg.kwargs['BLOCK_N']
        num_n = triton.cdiv(N, block_n)

        BLOCK_M_R = 64
        BLOCK_N_R = 64
        grid_r = (triton.cdiv(M, BLOCK_M_R),)
        reduce_kernel[grid_r](
            partial, out_flat,
            M, num_n,
            partial.stride(0), partial.stride(1),
            SCALE=scale,
            BLOCK_M=BLOCK_M_R,
            BLOCK_N=BLOCK_N_R,
            num_warps=4,
        )
        return out