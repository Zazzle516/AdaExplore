import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
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
    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc, input_precision="tf32")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(Bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
    acc = acc + bias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # weight is (N, K), used as B = weight.T -> (K, N)
    # Avoid transpose: pass strides accordingly
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # B = W^T: stride_bk = w.stride(1), stride_bn = w.stride(0)
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def gn_leaky_double_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    B, C, G,
    eps, negative_slope,
    CH_PER_G: tl.constexpr,
    GROUPS_PER_PROG: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    total_blocks_per_b = G // GROUPS_PER_PROG
    b = pid // total_blocks_per_b
    blk = pid % total_blocks_per_b
    g_start = blk * GROUPS_PER_PROG

    # Each program handles GROUPS_PER_PROG groups, each with CH_PER_G channels
    # Layout: 2D as [GROUPS_PER_PROG, CH_PER_G]
    offs_g = tl.arange(0, GROUPS_PER_PROG)
    offs_c = tl.arange(0, CH_PER_G)

    base = b * C + g_start * CH_PER_G
    # element index = grp_idx * CH_PER_G + ch_idx
    idx = offs_g[:, None] * CH_PER_G + offs_c[None, :]
    x = tl.load(x_ptr + base + idx).to(tl.float32)

    # Reduce per group (along axis=1, the channel axis)
    sum_x = tl.sum(x, axis=1)
    sum_x2 = tl.sum(x * x, axis=1)
    mean = sum_x / CH_PER_G
    var = sum_x2 / CH_PER_G - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    g_idx = g_start + offs_g
    gamma_idx = g_idx[:, None] * CH_PER_G + offs_c[None, :]
    gamma = tl.load(gamma_ptr + gamma_idx).to(tl.float32)
    beta = tl.load(beta_ptr + gamma_idx).to(tl.float32)

    y = (x - mean[:, None]) * rstd[:, None] * gamma + beta
    y = tl.where(y >= 0, y, y * negative_slope)
    y = y + y

    tl.store(out_ptr + base + idx, y)


def gn_leaky_double(x, gamma, beta, num_groups, eps, negative_slope):
    B, C = x.shape
    ch_per_g = C // num_groups
    # Pick GROUPS_PER_PROG so total threads per block is reasonable
    if ch_per_g <= 16:
        groups_per_prog = 16
    elif ch_per_g <= 32:
        groups_per_prog = 8
    elif ch_per_g <= 64:
        groups_per_prog = 4
    else:
        groups_per_prog = 1
    while num_groups % groups_per_prog != 0:
        groups_per_prog //= 2
    if groups_per_prog < 1:
        groups_per_prog = 1

    out = torch.empty_like(x)
    grid = (B * (num_groups // groups_per_prog),)
    total_elems = groups_per_prog * ch_per_g
    if total_elems <= 64:
        nw = 1
    elif total_elems <= 256:
        nw = 2
    elif total_elems <= 1024:
        nw = 4
    else:
        nw = 8
    gn_leaky_double_kernel[grid](
        x, gamma, beta, out,
        B, C, num_groups,
        eps, negative_slope,
        CH_PER_G=ch_per_g,
        GROUPS_PER_PROG=groups_per_prog,
        BLOCK_C=ch_per_g,
        num_warps=nw,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope

    def forward(self, x):
        x = x.contiguous()
        x = triton_linear(x, self.fc.weight, self.fc.bias)
        x = gn_leaky_double(
            x,
            self.gn.weight.contiguous(),
            self.gn.bias.contiguous(),
            self.num_groups,
            self.eps,
            self.negative_slope,
        )
        return x