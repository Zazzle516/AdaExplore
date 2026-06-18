import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_splitk_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    k_step = BLOCK_K * SPLIT_K
    n_iters = tl.cdiv(K - pid_k * BLOCK_K, k_step)

    cur_k = pid_k * BLOCK_K
    for _ in range(0, n_iters):
        k_mask = (cur_k + tl.arange(0, BLOCK_K)) < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += k_step * stride_ak
        b_ptrs += k_step * stride_bk
        cur_k += k_step

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask)


@triton.jit
def softmax_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_cols,
    stride_xm, stride_om,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_xm
    out_row = out_ptr + row * stride_om

    # Pass 1: max
    max_val = -float('inf')
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        v = tl.load(x_row + offs, mask=mask, other=-float('inf'))
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        v = v + b
        block_max = tl.max(v, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Pass 2: sum exp
    sum_val = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        v = tl.load(x_row + offs, mask=mask, other=-float('inf'))
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        v = v + b
        e = tl.exp(v - max_val)
        e = tl.where(mask, e, 0.0)
        sum_val += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_val
    # Pass 3: write
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        v = tl.load(x_row + offs, mask=mask, other=0.0)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        v = v + b
        e = tl.exp(v - max_val) * inv_sum
        tl.store(out_row + offs, e, mask=mask)


def triton_linear_splitk(x, weight, M, N, K):
    # weight: (N, K), we view as (K,N) via stride swap
    out = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
        meta['SPLIT_K'],
    )
    gemm_splitk_kernel[grid](
        x, weight, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


def triton_softmax_bias(x, bias):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = 2048
    grid = (M,)
    softmax_bias_kernel[grid](
        x, bias, out, N, x.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE, num_warps=8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        weight = self.matmul.weight.contiguous()
        bias = self.matmul.bias.contiguous()
        M, K = x.shape
        N = weight.shape[0]
        gemm_out = triton_linear_splitk(x, weight, M, N, K)
        if self.training:
            gemm_out = gemm_out + bias
            gemm_out = self.dropout(gemm_out)
            # softmax fallback
            return torch.softmax(gemm_out, dim=1)
        out = triton_softmax_bias(gemm_out, bias)
        return out