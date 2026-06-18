import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# GEMM + bias1 + bias2 + hardtanh + mish fused kernel
GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_act_kernel(
    A_ptr, B_ptr, b1_ptr, b2_ptr, C_ptr,
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
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
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

    # add biases
    b1 = tl.load(b1_ptr + offs_n, mask=mask_n, other=0.0)
    b2 = tl.load(b2_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b1[None, :] + b2[None, :]

    # hardtanh: clamp to [-1, 1]
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # mish: x * tanh(softplus(x)) = x * tanh(log(1+exp(x)))
    sp = tl.log(1.0 + tl.exp(acc))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    out = acc * tanh_sp

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


# GroupNorm kernel: per (row, group), compute mean/var across channels_per_group
@triton.jit
def group_norm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, C, G, CPG,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # over M*G
    row = pid // G
    grp = pid % G

    offs = tl.arange(0, BLOCK)
    mask = offs < CPG

    base = row * C + grp * CPG
    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(x, axis=0)
    mean = sum_x / CPG
    diff = x - mean
    diff = tl.where(mask, diff, 0.0)
    sum_sq = tl.sum(diff * diff, axis=0)
    var = sum_sq / CPG
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + grp * CPG + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + grp * CPG + offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * w + b
    tl.store(Y_ptr + base + offs, y, mask=mask)


def fused_gemm(x, weight, bias1, bias2):
    M, K = x.shape
    N, _ = weight.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_act_kernel[grid](
        x, weight, bias1, bias2, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


def group_norm_triton(x, weight, bias, num_groups, eps=1e-5):
    M, C = x.shape
    CPG = C // num_groups
    BLOCK = triton.next_power_of_2(CPG)
    out = torch.empty_like(x)
    grid = (M * num_groups,)
    group_norm_kernel[grid](
        x, weight, bias, out,
        M, C, num_groups, CPG,
        eps,
        BLOCK=BLOCK,
        num_warps=4 if BLOCK <= 256 else 8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        # Mirror reference parameter structure for state_dict compatibility
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous().cuda()
        # GEMM + bias1 + bias2 + hardtanh + mish
        y = fused_gemm(x, self.gemm.weight, self.gemm.bias, self.bias.view(-1))
        # GroupNorm
        out = group_norm_triton(y, self.groupnorm.weight, self.groupnorm.bias, self.num_groups, self.eps)
        return out