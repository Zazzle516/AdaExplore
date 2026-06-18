import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_kernel_splitk(
    a_ptr, b_ptr, bias_ptr, c_ptr,
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

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    k_step = BLOCK_K * SPLIT_K
    for k in range(pid_k * BLOCK_K, K, k_step):
        k_remaining = K - k
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (tl.arange(0, BLOCK_K)[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(tl.arange(0, BLOCK_K)[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += k_step * stride_ak
        b_ptrs += k_step * stride_bk

    if pid_k == 0:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def softmax_kernel(
    in_ptr, out_ptr, M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    row_ptr = in_ptr + row * stride_m
    out_row_ptr = out_ptr + row * stride_m

    max_val = -float('inf')
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_ptr + offs * stride_n, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = 0.0
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_ptr + offs * stride_n, mask=mask, other=-float('inf'))
        e = tl.exp(x - max_val)
        e = tl.where(mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_exp

    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_ptr + offs * stride_n, mask=mask, other=-float('inf'))
        e = tl.exp(x - max_val) * inv_sum
        tl.store(out_row_ptr + offs * stride_n, e, mask=mask)


def triton_linear_splitk(x, weight, bias, split_k=4):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    if split_k > 1:
        out = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    else:
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
        split_k,
    )
    gemm_bias_kernel_splitk[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        1, weight.stride(0),
        out.stride(0), out.stride(1),
        SPLIT_K=split_k,
    )
    return out


def triton_softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = 2048
    grid = (M,)
    softmax_kernel[grid](
        x, out, M, N,
        x.stride(0), x.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout_p = dropout_p

    def forward(self, x):
        x = x.cuda().contiguous()
        weight = self.matmul.weight.contiguous()
        bias = self.matmul.bias.contiguous()
        x = triton_linear_splitk(x, weight, bias, split_k=4)
        x = self.dropout(x)
        x = triton_softmax(x)
        return x