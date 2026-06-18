import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# ---------------- GEMM + bias + GELU ----------------
GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=2, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_gelu_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # bias add
    offs_n_real = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(Bias_ptr + offs_n_real, mask=offs_n_real < N, other=0.0)
    acc = acc + bias[None, :].to(tl.float32)

    # GELU (tanh approx)
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    k0 = 0.7978845608028654  # sqrt(2/pi)
    k1 = 0.044715
    x3 = acc * acc * acc
    inner = k0 * (acc + k1 * x3)
    # tanh via exp
    e2 = tl.exp(2.0 * inner)
    tanh_v = (e2 - 1.0) / (e2 + 1.0)
    out = 0.5 * acc * (1.0 + tanh_v)

    offs_m_real = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = C_ptr + offs_m_real[:, None] * stride_cm + offs_n_real[None, :] * stride_cn
    mask = (offs_m_real[:, None] < M) & (offs_n_real[None, :] < N)
    tl.store(c_ptrs, out, mask=mask)


# ---------------- Softmax (row-wise, dim=1) ----------------
@triton.jit
def softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    in_row_ptr = in_ptr + row * in_row_stride
    out_row_ptr = out_ptr + row * out_row_stride

    # pass 1: max
    max_val = -float('inf')
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(in_row_ptr + offs, mask=mask, other=-float('inf'))
        cur = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, cur)

    # pass 2: sum exp
    sum_val = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(in_row_ptr + offs, mask=mask, other=-float('inf'))
        e = tl.exp(x - max_val)
        e = tl.where(mask, e, 0.0)
        sum_val += tl.sum(e, axis=0)

    inv = 1.0 / sum_val
    # pass 3: write
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(in_row_ptr + offs, mask=mask, other=-float('inf'))
        y = tl.exp(x - max_val) * inv
        tl.store(out_row_ptr + offs, y, mask=mask)


def gemm_bias_gelu(x, w, b):
    # x: (M,K), w: (N,K) (linear weight), b: (N,)
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # B in matmul is w^T -> shape (K, N); we'll set strides accordingly
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_gelu_kernel[grid](
        x, w, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(1), w.stride(0),  # treat w as (K,N) with these strides
        out.stride(0), out.stride(1),
    )
    return out


def softmax_dim1(x):
    M, N = x.shape
    out = torch.empty_like(x)
    if N <= 1024:
        BLOCK = triton.next_power_of_2(N)
    elif N <= 4096:
        BLOCK = 2048
    else:
        BLOCK = 4096
    num_warps = 8 if BLOCK >= 2048 else 4
    softmax_kernel[(M,)](out, x, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK, num_warps=num_warps)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.linear.weight.contiguous()
        b = self.linear.bias.contiguous()
        y = gemm_bias_gelu(x, w, b)
        y = softmax_dim1(y)
        return y