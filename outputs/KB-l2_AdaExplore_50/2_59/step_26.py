import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 8}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def split_k_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    a_mask_m = offs_am < M
    b_mask_n = offs_bn < N

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_step = BLOCK_K * SPLIT_K
    for k in range(pid_k * BLOCK_K, K, k_step):
        k_remaining = K - k
        a = tl.load(a_ptrs, mask=(a_mask_m[:, None]) & (tl.arange(0, BLOCK_K)[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(b_mask_n[None, :]) & (tl.arange(0, BLOCK_K)[:, None] < k_remaining), other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += k_step * stride_ak
        b_ptrs += k_step * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
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
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(c_ptrs, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    x = x + bias[None, :]
    out = x * tl.sigmoid(x) * scaling_factor
    tl.store(c_ptrs, out, mask=mask)


def swish_linear(x, weight_t, bias, scaling_factor):
    # x: (M, K), weight_t: (K, N) -> compute x @ weight_t + bias
    M, K = x.shape
    N = weight_t.shape[1]
    x = x.contiguous()
    out = torch.zeros((M, N), device=x.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        META['SPLIT_K'],
    )
    split_k_gemm_kernel[grid](
        x, weight_t, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
    )

    BLOCK_M = 32
    BLOCK_N = 128
    grid2 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    bias_swish_scale_kernel[grid2](
        out, bias,
        M, N,
        out.stride(0), out.stride(1),
        scaling_factor,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        # Pre-transpose weight once for contiguous K-major layout
        with torch.no_grad():
            wt = self.matmul.weight.t().contiguous()
        self.register_buffer('weight_t', wt)

    def forward(self, x):
        return swish_linear(x, self.weight_t, self.matmul.bias, self.scaling_factor)