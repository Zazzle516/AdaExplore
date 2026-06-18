import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
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

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    row_mask = offs_am[:, None] < M
    col_mask = offs_bn[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=row_mask & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=col_mask & (offs_k[:, None] < k_remaining), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    # B = weight.T, so use weight as (N, K) with stride (K, 1) accessed as B[k,n] = weight[n,k]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # B[k,n] = weight[n,k]: stride_bk=stride along k=weight.stride(1)=1, stride_bn=weight.stride(0)=K
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def fused_gn_swish_mul_swish_kernel(
    X_ptr, GAMMA_ptr, BETA_ptr, MULW_ptr, OUT_ptr,
    N, C, G, CPG,
    eps,
    BLOCK: tl.constexpr,
):
    # one program per (sample, group)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    row_start = n * C + g * CPG
    offs = tl.arange(0, BLOCK)
    mask = offs < CPG

    x = tl.load(X_ptr + row_start + offs, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    mean = sum_x / CPG
    var = sum_x2 / CPG - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(GAMMA_ptr + g * CPG + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(BETA_ptr + g * CPG + offs, mask=mask, other=0.0).to(tl.float32)
    mulw = tl.load(MULW_ptr + g * CPG + offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * gamma + beta
    s1 = y * tl.sigmoid(y)
    z = s1 * mulw
    out = z * tl.sigmoid(z)

    tl.store(OUT_ptr + row_start + offs, out, mask=mask)


def fused_gn_swish_mul_swish(x, gamma, beta, mul_weight, num_groups, eps=1e-5):
    N, C = x.shape
    G = num_groups
    CPG = C // G
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(CPG)
    grid = (N * G,)
    fused_gn_swish_mul_swish_kernel[grid](
        x, gamma, beta, mul_weight, out,
        N, C, G, CPG, eps,
        BLOCK=BLOCK,
        num_warps=4 if BLOCK >= 128 else 2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.gemm.weight.contiguous()
        b = self.gemm.bias.contiguous()
        y = triton_linear(x, w, b)
        out = fused_gn_swish_mul_swish(
            y,
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            self.multiply_weight.contiguous(),
            self.num_groups,
            self.group_norm.eps,
        )
        return out