import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def split_k_gemm_kernel(
    x_ptr, w_ptr, partial_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pk, stride_pm, stride_pn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    k_step = BLOCK_K * SPLIT_K
    num_iters = tl.cdiv(K - pid_k * BLOCK_K, k_step)

    for i in range(0, num_iters):
        cur_k = pid_k * BLOCK_K + i * k_step
        mask_k = (tl.arange(0, BLOCK_K) + cur_k) < K
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w, allow_tf32=True)
        x_ptrs += k_step * stride_xk
        w_ptrs += k_step * stride_wk

    p_ptrs = partial_ptr + pid_k * stride_pk + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
    tl.store(p_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def epilogue_kernel(
    partial_ptr, b_ptr, c_ptr, out_ptr,
    M, N, SPLIT_K,
    stride_pk, stride_pm, stride_pn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, SPLIT_K):
        p_ptrs = partial_ptr + k * stride_pk + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
        v = tl.load(p_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        acc += v

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    c = tl.load(c_ptr)
    acc = tl.minimum(acc, c) - c

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features
        w = self.linear.weight
        b = self.linear.bias
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # Allocate partial buffer; size depends on autotune-selected SPLIT_K, so allocate max
        MAX_SPLIT_K = 8
        partial = torch.empty((MAX_SPLIT_K, M, N), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
            meta['SPLIT_K'],
        )
        split_k_gemm_kernel[grid](
            x, w, partial,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            partial.stride(0), partial.stride(1), partial.stride(2),
        )

        # Find the SPLIT_K used by best config
        best_config = split_k_gemm_kernel.best_config
        SPLIT_K = best_config.kwargs['SPLIT_K']

        BLOCK_M_EPI = 32
        BLOCK_N_EPI = 128
        epi_grid = (triton.cdiv(M, BLOCK_M_EPI), triton.cdiv(N, BLOCK_N_EPI))
        epilogue_kernel[epi_grid](
            partial, b, self.constant, out,
            M, N, SPLIT_K,
            partial.stride(0), partial.stride(1), partial.stride(2),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M_EPI, BLOCK_N=BLOCK_N_EPI,
        )
        return out