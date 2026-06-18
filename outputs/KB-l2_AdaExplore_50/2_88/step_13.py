import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
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
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    acc = acc + bias[None, :]
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear(x, weight, bias):
    # x: (M, K), weight: (N, K), bias: (N,)
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # weight.T effectively: weight is (N,K), we use it as (K,N) so stride_bk=weight.stride(1)=1, stride_bn=weight.stride(0)=K
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def fused_gn_swish_mul_swish_kernel(
    x_ptr, gamma_ptr, beta_ptr, mw_ptr, out_ptr,
    C, G, CH_PER_G,
    eps,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_start = pid_g * CH_PER_G
    offs = tl.arange(0, BLOCK)
    mask = offs < CH_PER_G

    chan_off = group_start + offs
    row_base = pid_n * C

    x = tl.load(x_ptr + row_base + chan_off, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = sum_x / CH_PER_G
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / CH_PER_G
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + chan_off, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + chan_off, mask=mask, other=0.0).to(tl.float32)
    mw = tl.load(mw_ptr + chan_off, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * gamma + beta
    # swish
    s1 = y * tl.sigmoid(y)
    # multiply weight
    z = s1 * mw
    # swish again
    out = z * tl.sigmoid(z)

    tl.store(out_ptr + row_base + chan_off, out, mask=mask)


def fused_gn_swish_mul_swish(x, gamma, beta, mw, num_groups, eps=1e-5):
    N, C = x.shape
    assert C % num_groups == 0
    CH_PER_G = C // num_groups
    out = torch.empty_like(x)
    # next power of 2 for BLOCK
    BLOCK = triton.next_power_of_2(CH_PER_G)
    grid = (N, num_groups)
    fused_gn_swish_mul_swish_kernel[grid](
        x, gamma, beta, mw, out,
        C, num_groups, CH_PER_G,
        eps,
        BLOCK=BLOCK,
        num_warps=4 if BLOCK <= 128 else 8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        x = triton_linear(x.contiguous(), self.gemm.weight, self.gemm.bias)
        out = fused_gn_swish_mul_swish(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            self.num_groups,
            self.eps,
        )
        return out