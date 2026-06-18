import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K', 'SPLIT_K'],
)
@triton.jit
def split_k_gemm_kernel(
    x_ptr, w_ptr, partial_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pm, stride_pn, stride_ps,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * K_per_split
    k_end = tl.minimum(k_start + K_per_split, K)
    n_iters = tl.cdiv(k_end - k_start, BLOCK_K)

    for ki in range(0, n_iters):
        k_offs = k_start + ki * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < k_end
        x = tl.load(x_ptrs + ki * BLOCK_K * stride_xk,
                    mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs + ki * BLOCK_K * stride_wk,
                    mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)

    out_ptrs = partial_ptr + (offs_m[:, None] * stride_pm
                              + offs_n[None, :] * stride_pn
                              + pid_k * stride_ps)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def reduce_epilogue_kernel(
    partial_ptr, b_ptr, out_ptr,
    M, N,
    stride_pm, stride_pn, stride_ps,
    stride_om, stride_on,
    SPLIT_K: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    base = partial_ptr + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn
    for s in range(0, SPLIT_K):
        p = tl.load(base + s * stride_ps,
                    mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        acc += p

    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    acc = acc * tl.sigmoid(acc) * SCALE

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def linear_swish_scale(x, weight, bias, scale):
    M, K = x.shape
    N = weight.shape[0]
    SPLIT_K = 8

    partial = torch.empty((M, N, SPLIT_K), device=x.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
        SPLIT_K,
    )
    split_k_gemm_kernel[grid](
        x, weight, partial,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        partial.stride(0), partial.stride(1), partial.stride(2),
        SPLIT_K=SPLIT_K,
    )

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M = 32
    BLOCK_N = 128
    grid2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    reduce_epilogue_kernel[grid2](
        partial, bias, out,
        M, N,
        partial.stride(0), partial.stride(1), partial.stride(2),
        out.stride(0), out.stride(1),
        SPLIT_K=SPLIT_K,
        SCALE=float(scale),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = x.contiguous()
        w = self.matmul.weight.contiguous()
        b = self.matmul.bias.contiguous()
        return linear_swish_scale(x, w, b, self.scaling_factor)