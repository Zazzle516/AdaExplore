import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_kernel_fp16(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def softmax_kernel_2pass(
    in_ptr, out_ptr, M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    row_ptr = in_ptr + row * stride_m
    out_row_ptr = out_ptr + row * stride_m

    # First pass: compute max and sum simultaneously (online)
    max_val = -float('inf')
    sum_exp = 0.0
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_ptr + offs * stride_n, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, block_max)
        # rescale prior sum
        sum_exp = sum_exp * tl.exp(max_val - new_max)
        e = tl.exp(x - new_max)
        e = tl.where(mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)
        max_val = new_max

    inv_sum = 1.0 / sum_exp

    # Second pass: write normalized
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_ptr + offs * stride_n, mask=mask, other=-float('inf'))
        e = tl.exp(x - max_val) * inv_sum
        tl.store(out_row_ptr + offs * stride_n, e, mask=mask)


def triton_linear_fp16(x_fp16, weight_fp16, bias):
    M, K = x_fp16.shape
    N, K2 = weight_fp16.shape
    assert K == K2
    out = torch.empty((M, N), device=x_fp16.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_bias_kernel_fp16[grid](
        x_fp16, weight_fp16, bias, out,
        M, N, K,
        x_fp16.stride(0), x_fp16.stride(1),
        1, weight_fp16.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


def triton_softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = 2048
    grid = (M,)
    softmax_kernel_2pass[grid](
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
        # Pre-cast weight to fp16 for tensor core utilization
        self._weight_fp16 = None

    def _get_weight_fp16(self):
        if self._weight_fp16 is None or self._weight_fp16.shape != self.matmul.weight.shape:
            self._weight_fp16 = self.matmul.weight.detach().to(torch.float16).contiguous().cuda()
        return self._weight_fp16

    def forward(self, x):
        x = x.cuda().contiguous()
        x_fp16 = x.to(torch.float16)
        weight_fp16 = self._get_weight_fp16()
        bias = self.matmul.bias.contiguous().cuda()
        out = triton_linear_fp16(x_fp16, weight_fp16, bias)
        out = self.dropout(out)
        out = triton_softmax(out)
        return out