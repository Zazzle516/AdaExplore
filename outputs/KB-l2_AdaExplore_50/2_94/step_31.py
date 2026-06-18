import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# GEMM + bias + hardtanh + mish fused kernel
GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_hardtanh_mish_kernel(
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
        mask_k = offs_k[None, :] < (K - k * BLOCK_K)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k, other=0.0)
        mask_k2 = offs_k[:, None] < (K - k * BLOCK_K)
        b = tl.load(b_ptrs, mask=mask_k2 & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # bias add
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # hardtanh: clamp to [-1, 1]
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # mish: x * tanh(softplus(x))
    # stable softplus: max(x,0) + log(1 + exp(-|x|))
    abs_acc = tl.abs(acc)
    sp = tl.maximum(acc, 0.0) + tl.log(1.0 + tl.exp(-abs_acc))
    # stable tanh: tanh(sp) where sp >= 0 always
    e_neg = tl.exp(-2.0 * sp)
    th = (1.0 - e_neg) / (1.0 + e_neg)
    out = acc * th

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


# GroupNorm kernel: each program handles one (batch, group)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=['CPG'],
)
@triton.jit
def group_norm_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    N, C, G, CPG,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    # offsets within the group
    row_start = n * C + g * CPG
    offs = tl.arange(0, BLOCK)
    mask = offs < CPG

    x = tl.load(X_ptr + row_start + offs, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(x, axis=0)
    mean = sum_x / CPG
    diff = x - mean
    diff = tl.where(mask, diff, 0.0)
    sum_sq = tl.sum(diff * diff, axis=0)
    var = sum_sq / CPG
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + g * CPG + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + g * CPG + offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * w + b
    tl.store(Y_ptr + row_start + offs, y, mask=mask)


def fused_gemm(x, weight, bias_lin, bias_extra):
    # x: (M, K), weight: (N, K), bias_lin: (N,), bias_extra: (N,)
    M, K = x.shape
    N = weight.shape[0]
    bias = bias_lin + bias_extra  # combine into one bias
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # B = weight.T  -> (K, N), but we use stride-based access
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_hardtanh_mish_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # B = weight.T
        out.stride(0), out.stride(1),
    )
    return out


def group_norm_forward(x, weight, bias, num_groups, eps=1e-5):
    N, C = x.shape
    CPG = C // num_groups
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(CPG)
    grid = (N * num_groups,)
    group_norm_kernel[grid](
        x, out, weight, bias,
        N, C, num_groups, CPG,
        eps,
        BLOCK=BLOCK,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        # match the reference initialization
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def forward(self, x):
        x = x.contiguous()
        weight = self.gemm.weight
        if not weight.is_contiguous():
            weight = weight.contiguous()
        bias_lin = self.gemm.bias.contiguous()
        bias_extra = self.bias.contiguous()

        y = fused_gemm(x, weight, bias_lin, bias_extra)

        out = group_norm_forward(
            y,
            self.groupnorm.weight.contiguous(),
            self.groupnorm.bias.contiguous(),
            self.num_groups,
            float(self.groupnorm.eps),
        )
        return out