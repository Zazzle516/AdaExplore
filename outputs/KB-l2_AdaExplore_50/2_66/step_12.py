import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_dropout_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    seed, dropout_p, scale,
    APPLY_DROPOUT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_DROPOUT:
        # generate random mask
        offs = offs_m[:, None] * N + offs_n[None, :]
        rnd = tl.rand(seed, offs)
        keep = rnd >= dropout_p
        acc = tl.where(keep, acc * scale, 0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def softmax_online_kernel(
    in_ptr, out_ptr, M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    row_ptr = in_ptr + row * stride_m
    out_row_ptr = out_ptr + row * stride_m

    # Online softmax: single pass for max+sum, then write
    running_max = -float('inf')
    running_sum = 0.0

    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_ptr + offs * stride_n, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, block_max)
        # correct previous sum
        running_sum = running_sum * tl.exp(running_max - new_max)
        e = tl.exp(x - new_max)
        e = tl.where(mask, e, 0.0)
        running_sum += tl.sum(e, axis=0)
        running_max = new_max

    inv_sum = 1.0 / running_sum

    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_ptr + offs * stride_n, mask=mask, other=-float('inf'))
        e = tl.exp(x - running_max) * inv_sum
        tl.store(out_row_ptr + offs * stride_n, e, mask=mask)


def triton_linear_dropout(x, weight, bias, dropout_p, training):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    apply_dropout = training and dropout_p > 0.0
    seed = torch.randint(0, 2**31 - 1, (1,), device=x.device).item() if apply_dropout else 0
    scale = 1.0 / (1.0 - dropout_p) if apply_dropout else 1.0

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_bias_dropout_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        1, weight.stride(0),
        out.stride(0), out.stride(1),
        seed, dropout_p, scale,
        APPLY_DROPOUT=apply_dropout,
    )
    return out


def triton_softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = 4096
    grid = (M,)
    softmax_online_kernel[grid](
        x, out, M, N,
        x.stride(0), x.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=16,
        num_stages=2,
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
        x = triton_linear_dropout(x, weight, bias, self.dropout_p, self.training)
        x = triton_softmax(x)
        return x