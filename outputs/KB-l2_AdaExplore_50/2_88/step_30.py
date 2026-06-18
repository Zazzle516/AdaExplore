import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_kernel(
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = mask_m[:, None] & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & mask_n[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, _ = weight.shape
    x = x.contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    # weight is (N, K) row-major: weight.stride(0)=K, weight.stride(1)=1
    # We need B as (K, N) logical with stride_bk and stride_bn:
    # stride_bk = stride along K = weight.stride(1) = 1
    # stride_bn = stride along N = weight.stride(0) = K
    gemm_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def fused_gn_swish_mul_swish_kernel(
    X_ptr, GAMMA_ptr, BETA_ptr, MW_ptr, OUT_ptr,
    N, C, G, CPG,
    eps,
    BLOCK_CPG: tl.constexpr,
):
    # one program per (batch, group)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs = tl.arange(0, BLOCK_CPG)
    mask = offs < CPG

    # channel index within full C
    c_idx = pid_g * CPG + offs

    x_ptrs = X_ptr + pid_n * C + c_idx
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # compute mean and var
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = sum_x / CPG
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / CPG
    rstd = tl.rsqrt(var + eps)

    gamma = tl.load(GAMMA_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(BETA_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    mw = tl.load(MW_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * gamma + beta
    # swish 1
    s1 = tl.sigmoid(y)
    y1 = y * s1
    # multiply
    y2 = y1 * mw
    # swish 2
    s2 = tl.sigmoid(y2)
    out = y2 * s2

    tl.store(OUT_ptr + pid_n * C + c_idx, out, mask=mask)


def fused_gn_swish_mul_swish(x, gamma, beta, mw, num_groups, eps=1e-5):
    N, C = x.shape
    G = num_groups
    CPG = C // G
    out = torch.empty_like(x)
    # next power of 2 for CPG
    BLOCK_CPG = triton.next_power_of_2(CPG)
    grid = (N, G)
    fused_gn_swish_mul_swish_kernel[grid](
        x, gamma, beta, mw, out,
        N, C, G, CPG,
        eps,
        BLOCK_CPG=BLOCK_CPG,
        num_warps=4 if BLOCK_CPG <= 128 else 8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous().cuda()
        y = triton_linear(x, self.gemm.weight, self.gemm.bias)
        out = fused_gn_swish_mul_swish(
            y,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            self.num_groups,
            self.eps,
        )
        return out