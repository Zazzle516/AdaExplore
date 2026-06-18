import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(k_mask[:, None]) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = tl.load(C_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + c[None, :]
    out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, _ = weight.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return out


@triton.jit
def gn_min_kernel(
    X_ptr, Gamma_ptr, Beta_ptr, Out_ptr,
    M, C, G, CPG,
    eps,
    BLOCK_CPG: tl.constexpr, NUM_GROUPS: tl.constexpr,
):
    # one program per (row, group)
    row = tl.program_id(0)
    g = tl.program_id(1)

    offs = tl.arange(0, BLOCK_CPG)
    mask = offs < CPG
    x_ptrs = X_ptr + row * C + g * CPG + offs
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # mean and var
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / CPG
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / CPG
    rstd = 1.0 / tl.sqrt(var + eps)

    # load affine
    gamma = tl.load(Gamma_ptr + g * CPG + offs, mask=mask, other=0.0)
    beta = tl.load(Beta_ptr + g * CPG + offs, mask=mask, other=0.0)

    y = xc * rstd * gamma + beta

    # min over this group's elements (masked positions = +inf)
    y_masked = tl.where(mask, y, float('inf'))
    group_min = tl.min(y_masked, axis=0)

    # Atomically reduce min across groups for this row
    tl.atomic_min(Out_ptr + row, group_min)


def triton_gn_min(x, gamma, beta, num_groups, eps):
    M, C = x.shape
    CPG = C // num_groups
    # next power of 2 for BLOCK_CPG
    BLOCK_CPG = triton.next_power_of_2(CPG)
    out = torch.full((M,), float('inf'), device=x.device, dtype=x.dtype)
    grid = (M, num_groups)
    gn_min_kernel[grid](
        x, gamma, beta, out,
        M, C, num_groups, CPG,
        eps,
        BLOCK_CPG=BLOCK_CPG, NUM_GROUPS=num_groups,
        num_warps=1,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.num_groups = num_groups
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        W = self.gemm.weight.contiguous()
        b = self.gemm.bias.contiguous()
        y = triton_linear(x, W, b)  # (M, N)

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps

        m = triton_gn_min(y, gamma, beta, self.num_groups, eps)  # (M,)
        # reshape to (M, 1) to match torch.min(..., keepdim=True) shape
        m = m.view(-1, 1)
        out = m + self.bias  # broadcast: bias is (1, out_features, 1, 1)
        return out