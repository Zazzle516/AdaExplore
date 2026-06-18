import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def splitk_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_mn = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    K_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * K_per_split
    k_end = tl.minimum(k_start + K_per_split, K)

    offs_k0 = k_start + tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k0[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k0[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_iters = tl.cdiv(k_end - k_start, BLOCK_K)
    for k in range(0, num_iters):
        k_cur = k_start + k * BLOCK_K
        k_remaining = k_end - k_cur
        mask_k = tl.arange(0, BLOCK_K) < k_remaining
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]

    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=c_mask)


@triton.jit
def bias_swish_scale_kernel(
    C_ptr, bias_ptr,
    M, N,
    stride_cm, stride_cn,
    scaling_factor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    val = tl.load(c_ptrs, mask=mask, other=0.0)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    val = val + bias[None, :]
    sig = tl.sigmoid(val)
    out = val * sig * scaling_factor
    tl.store(c_ptrs, out, mask=mask)


def swish_linear(x, weight, bias, scaling_factor):
    M, K = x.shape
    N = weight.shape[0]
    x = x.contiguous()
    B = weight.t().contiguous()

    out = torch.zeros((M, N), device=x.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        META['SPLIT_K'],
    )
    splitk_gemm_kernel[grid](
        x, B, out,
        M, N, K,
        x.stride(0), x.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
    )

    # apply bias + swish + scale
    BLOCK_M2 = 32
    BLOCK_N2 = 256
    grid2 = (triton.cdiv(M, BLOCK_M2), triton.cdiv(N, BLOCK_N2))
    bias_swish_scale_kernel[grid2](
        out, bias,
        M, N,
        out.stride(0), out.stride(1),
        scaling_factor,
        BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        return swish_linear(x, self.matmul.weight, self.matmul.bias, self.scaling_factor)