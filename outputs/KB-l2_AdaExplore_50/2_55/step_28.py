import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_pool_kernel(
    x_ptr, wt_ptr, b_ptr, partial_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_pt, stride_pm,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Block swizzle for L2 reuse
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # x: [M, K], wt: [K, N] (transposed weight, contiguous along N)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = wt_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
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

    # Store partial sum to partial[pid_n, offs_m]
    p_ptrs = partial_ptr + pid_n * stride_pt + offs_m * stride_pm
    tl.store(p_ptrs, partial, mask=mask_m)


@triton.jit
def reduce_scale_kernel(
    partial_ptr, out_ptr,
    M, NTILES,
    stride_pt, stride_pm,
    scale,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    mask_t = offs_t < NTILES
    p_ptrs = partial_ptr + offs_t * stride_pt + pid_m * stride_pm
    vals = tl.load(p_ptrs, mask=mask_t, other=0.0)
    s = tl.sum(vals, axis=0) * scale
    tl.store(out_ptr + pid_m, s)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)

        lin = nn.Linear(in_features, out_features)
        # Pre-transpose weight to (K, N) contiguous so inner B-load is contiguous along N
        weight_t = lin.weight.detach().cuda().t().contiguous()  # (K, N)
        self.weight_t = nn.Parameter(weight_t)
        self.bias = nn.Parameter(lin.bias.detach().cuda().contiguous())

    def forward(self, x):
        x = x.cuda().contiguous()
        B = x.shape[0]
        K = self.in_features
        N = self.out_features
        KS = self.kernel_size

        # Worst-case num_n_tiles uses smallest BLOCK_N in configs (64)
        MIN_BLOCK_N = 64
        max_tiles = (N + MIN_BLOCK_N - 1) // MIN_BLOCK_N
        partial = torch.empty((max_tiles, B), device=x.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(B, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )

        fused_gemm_pool_kernel[grid](
            x, self.weight_t, self.bias, partial,
            B, N, K,
            x.stride(0), x.stride(1),
            self.weight_t.stride(0), self.weight_t.stride(1),
            partial.stride(0), partial.stride(1),
            KERNEL_SIZE=KS,
        )

        best = fused_gemm_pool_kernel.best_config
        chosen_BLOCK_N = best.kwargs['BLOCK_N']
        num_n_tiles = (N + chosen_BLOCK_N - 1) // chosen_BLOCK_N

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