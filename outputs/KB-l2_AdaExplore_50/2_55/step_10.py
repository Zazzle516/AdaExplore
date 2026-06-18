import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_partial_kernel(
    X_ptr, W_ptr, B_ptr, Partial_ptr,
    M, N, K,
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

    out_ptrs = Partial_ptr + offs_m * stride_pm + pid_n * stride_pn
    tl.store(out_ptrs, partial, mask=mask_m)


@triton.jit
def reduce_kernel(
    Partial_ptr, Out_ptr,
    M, NTILES,
    stride_pm, stride_pn,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < NTILES
    ptrs = Partial_ptr + pid_m * stride_pm + offs_n * stride_pn
    v = tl.load(ptrs, mask=mask_n, other=0.0)
    s = tl.sum(v, axis=0) * SCALE
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

        # We'll allocate partial buffer based on a fixed BLOCK_N from autotune.
        # Use a placeholder size: max BLOCK_N from configs = 256
        # We need to know BLOCK_N to compute NTILES. Use a wrapper: allocate maximally for worst-case
        # Strategy: pre-decide BLOCK_N choices come from autotune; we don't know which.
        # Solution: allocate partial size large enough for smallest BLOCK_N (64), so NTILES_MAX = N/64
        MAX_NTILES = N // 64
        partial = torch.empty((M, MAX_NTILES), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        compiled = fused_partial_kernel[grid](
            x, W, B, partial,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            partial.stride(0), partial.stride(1),
            KERNEL_SIZE=self.kernel_size,
        )

        # Determine actual BLOCK_N used
        BLOCK_N = compiled.metadata.get('BLOCK_N', None) if hasattr(compiled, 'metadata') else None
        # Fallback: read from best_config
        if BLOCK_N is None:
            BLOCK_N = fused_partial_kernel.best_config.kwargs['BLOCK_N']
        NTILES = N // BLOCK_N

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        # Next power of two >= NTILES
        BLOCK_REDUCE = 1
        while BLOCK_REDUCE < NTILES:
            BLOCK_REDUCE *= 2

        reduce_kernel[(M,)](
            partial, out,
            M, NTILES,
            partial.stride(0), partial.stride(1),
            SCALE=self.scale_factor,
            BLOCK_N=BLOCK_REDUCE,
        )

        return out