import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_bias_tf32_kernel(
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
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :].to(tl.float32)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def fused_gn_swish_mul_swish_kernel(
    x_ptr, gamma_ptr, beta_ptr, mw_ptr, out_ptr,
    M, N,
    G: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    eps: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_start = pid_g * GROUPS_PER_BLOCK
    g_off = tl.arange(0, GROUPS_PER_BLOCK)
    c_off = tl.arange(0, GROUP_SIZE)

    col_idx = (g_start + g_off)[:, None] * GROUP_SIZE + c_off[None, :]
    row_off = pid_m * N
    ptrs = x_ptr + row_off + col_idx

    g_mask = (g_start + g_off) < G
    mask = g_mask[:, None]

    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=1) / GROUP_SIZE
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / GROUP_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)

    x_norm = xc * rstd[:, None]

    gamma = tl.load(gamma_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
    mw = tl.load(mw_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)

    y = x_norm * gamma + beta
    y = y * tl.sigmoid(y)
    y = y * mw
    y = y * tl.sigmoid(y)

    tl.store(out_ptr + row_off + col_idx, y, mask=mask)


def triton_gemm_bias(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_tf32_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


def fused_gn_swish_mul_swish(x, gamma, beta, mw, num_groups, eps=1e-5):
    M, N = x.shape
    GROUP_SIZE = N // num_groups
    G = num_groups
    out = torch.empty_like(x)

    if GROUP_SIZE <= 32:
        GROUPS_PER_BLOCK = 8
    elif GROUP_SIZE <= 64:
        GROUPS_PER_BLOCK = 4
    elif GROUP_SIZE <= 128:
        GROUPS_PER_BLOCK = 2
    else:
        GROUPS_PER_BLOCK = 1

    num_g_blocks = (G + GROUPS_PER_BLOCK - 1) // GROUPS_PER_BLOCK
    grid = (M, num_g_blocks)

    elements_per_block = GROUPS_PER_BLOCK * GROUP_SIZE
    if elements_per_block >= 512:
        num_warps = 8
    elif elements_per_block <= 64:
        num_warps = 2
    else:
        num_warps = 4

    fused_gn_swish_mul_swish_kernel[grid](
        x, gamma, beta, mw, out,
        M, N,
        G=G,
        GROUP_SIZE=GROUP_SIZE,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        eps=eps,
        num_warps=num_warps,
        num_stages=2,
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
        x = x.contiguous()
        w = self.gemm.weight.contiguous()
        b = self.gemm.bias.contiguous()
        y = triton_gemm_bias(x, w, b)
        out = fused_gn_swish_mul_swish(
            y,
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            self.multiply_weight.contiguous(),
            self.num_groups,
            self.eps,
        )
        return out