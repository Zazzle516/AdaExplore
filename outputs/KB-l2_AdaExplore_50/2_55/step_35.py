import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_pool_partial_kernel(
    X_ptr, W_ptr, B_ptr, Partial_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pm, stride_pn,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        w = tl.load(w_ptrs)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(B_ptr + offs_n)
    acc = acc + b[None, :]

    POOLED: tl.constexpr = BLOCK_N // KERNEL_SIZE
    acc_r = tl.reshape(acc, (BLOCK_M, POOLED, KERNEL_SIZE))
    pooled = tl.max(acc_r, axis=2)
    partial = tl.sum(pooled, axis=1)

    p_ptrs = Partial_ptr + offs_m * stride_pm + pid_n * stride_pn
    tl.store(p_ptrs, partial, mask=mask_m)


@triton.jit
def reduce_partial_kernel(
    Partial_ptr, Out_ptr,
    M, NTILES,
    stride_pm, stride_pn,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < NTILES
    p_ptrs = Partial_ptr + pid_m * stride_pm + offs_n * stride_pn
    vals = tl.load(p_ptrs, mask=mask_n, other=0.0)
    s = tl.sum(vals, axis=0)
    s = s * SCALE
    tl.store(Out_ptr + pid_m, s)


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
        W = self.matmul.weight.contiguous().cuda()
        B = self.matmul.bias.contiguous().cuda()

        M, K = x.shape
        N = self.out_features

        # Workspace for partial sums per (row, N-tile)
        # We don't know BLOCK_N statically until autotune picks; allocate worst case for smallest BLOCK_N.
        # Use a fixed partial allocation by overestimating: use min BLOCK_N = 128.
        # To be safe, allocate based on actual chosen BLOCK_N via passing N tiles dynamically.
        # We'll allocate enough for smallest config (BLOCK_N=128 -> 256 tiles).
        max_ntiles = (N + 127) // 128
        partial = torch.empty((M, max_ntiles), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        )

        fused_gemm_pool_partial_kernel[grid](
            x, W, B, partial,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            partial.stride(0), partial.stride(1),
            KERNEL_SIZE=self.kernel_size,
            GROUP_M=8,
        )

        # Determine actual NTILES used = N / BLOCK_N from chosen config
        # Best meta is in fused_gemm_pool_partial_kernel.best_config
        best = fused_gemm_pool_partial_kernel.best_config
        block_n = best.kwargs['BLOCK_N']
        ntiles = (N + block_n - 1) // block_n

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        # next pow 2 >= ntiles
        BLOCK_N_RED = 1
        while BLOCK_N_RED < ntiles:
            BLOCK_N_RED *= 2

        reduce_partial_kernel[(M,)](
            partial, out,
            M, ntiles,
            partial.stride(0), partial.stride(1),
            SCALE=self.scale_factor,
            BLOCK_N=BLOCK_N_RED,
        )

        return out