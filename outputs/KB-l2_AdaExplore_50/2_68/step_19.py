import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _splitk_gemm_kernel(
    x_ptr, w_ptr, partial_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pm, stride_pn, stride_ps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Split K: each program handles K range [pid_k * K_per_split, (pid_k+1) * K_per_split)
    K_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * K_per_split
    k_end = tl.minimum(k_start + K_per_split, K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k[None, :]) * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + (k_start + offs_k[:, None]) * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    num_iters = tl.cdiv(k_end - k_start, BLOCK_K)
    for k in range(0, num_iters):
        cur_k = k_start + k * BLOCK_K
        k_remaining = k_end - cur_k
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    p_ptrs = partial_ptr + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn + pid_k * stride_ps
    tl.store(p_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _reduce_epilogue_kernel(
    partial_ptr, b_ptr, c_ptr, out_ptr,
    M, N, SPLIT_K,
    stride_pm, stride_pn, stride_ps,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in range(0, SPLIT_K):
        p_ptrs = partial_ptr + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn + s * stride_ps
        p = tl.load(p_ptrs, mask=mask, other=0.0)
        acc += p

    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b[None, :]

    c = tl.load(c_ptr)
    acc = tl.minimum(acc, c) - c

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features
        W = self.linear.weight  # (N, K)
        b = self.linear.bias    # (N,)

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # We need to know SPLIT_K to allocate partial buffer. We'll pick a fixed
        # max SPLIT_K and have the autotuner use values up to that.
        MAX_SPLIT_K = 8
        partial = torch.empty((MAX_SPLIT_K, M, N), device=x.device, dtype=torch.float32)

        def grid(meta):
            return (
                triton.cdiv(M, meta['BLOCK_M']),
                triton.cdiv(N, meta['BLOCK_N']),
                meta['SPLIT_K'],
            )

        _splitk_gemm_kernel[grid](
            x, W, partial,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            partial.stride(1), partial.stride(2), partial.stride(0),
        )

        # get the selected SPLIT_K from the best config
        best_config = _splitk_gemm_kernel.best_config
        split_k = best_config.kwargs['SPLIT_K']

        BLOCK_M_RED = 32 if M >= 32 else M
        BLOCK_N_RED = 128
        reduce_grid = (triton.cdiv(M, BLOCK_M_RED), triton.cdiv(N, BLOCK_N_RED))
        _reduce_epilogue_kernel[reduce_grid](
            partial, b, self.constant, out,
            M, N, split_k,
            partial.stride(1), partial.stride(2), partial.stride(0),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M_RED, BLOCK_N=BLOCK_N_RED,
            num_warps=4,
        )
        return out