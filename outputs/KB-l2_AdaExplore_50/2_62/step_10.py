import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_tf32_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
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
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, input_precision="tf32")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    c = accumulator + bias[None, :]
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_linear_tf32(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
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


@triton.jit
def gn_leaky_double_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    B, C, G,
    eps, negative_slope,
    CH_PER_G: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    GROUPS_PER_ROW: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    # Each program handles ROWS_PER_PROG rows, one full row of C channels
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG

    offs_ch = tl.arange(0, CH_PER_G)  # channels within a group

    for r in tl.static_range(ROWS_PER_PROG):
        row = row_start + r
        if row < B:
            row_base = row * C
            # Process each group in this row
            for g in range(0, G):
                base = row_base + g * CH_PER_G
                x = tl.load(x_ptr + base + offs_ch).to(tl.float32)
                sum_x = tl.sum(x, axis=0)
                sum_x2 = tl.sum(x * x, axis=0)
                mean = sum_x / CH_PER_G
                var = sum_x2 / CH_PER_G - mean * mean
                rstd = 1.0 / tl.sqrt(var + eps)

                gamma = tl.load(gamma_ptr + g * CH_PER_G + offs_ch).to(tl.float32)
                beta = tl.load(beta_ptr + g * CH_PER_G + offs_ch).to(tl.float32)

                y = (x - mean) * rstd * gamma + beta
                y = tl.where(y >= 0, y, y * negative_slope)
                y = y + y
                tl.store(out_ptr + base + offs_ch, y)


def gn_leaky_double(x, gamma, beta, num_groups, eps, negative_slope):
    B, C = x.shape
    ch_per_g = C // num_groups
    out = torch.empty_like(x)

    # one program per row; row handles all groups
    ROWS_PER_PROG = 1
    grid = (triton.cdiv(B, ROWS_PER_PROG),)
    BLOCK = 1
    while BLOCK < ch_per_g:
        BLOCK *= 2

    gn_leaky_double_kernel[grid](
        x, gamma, beta, out,
        B, C, num_groups,
        eps, negative_slope,
        CH_PER_G=BLOCK,
        ROWS_PER_PROG=ROWS_PER_PROG,
        GROUPS_PER_ROW=num_groups,
        BLOCK_CHUNK=BLOCK,
        num_warps=1,
    )
    return out


@triton.jit
def gn_leaky_double_simple_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    B, C, G, CH_PER_G: tl.constexpr,
    eps, negative_slope,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // G
    g = pid % G

    offs = tl.arange(0, BLOCK)
    mask = offs < CH_PER_G

    base = b * C + g * CH_PER_G
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    mean = sum_x / CH_PER_G
    var = sum_x2 / CH_PER_G - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + g * CH_PER_G + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + g * CH_PER_G + offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * gamma + beta
    y = tl.where(y >= 0, y, y * negative_slope)
    y = y + y

    tl.store(out_ptr + base + offs, y, mask=mask)


def gn_leaky_double_simple(x, gamma, beta, num_groups, eps, negative_slope):
    B, C = x.shape
    ch_per_g = C // num_groups
    BLOCK = 1
    while BLOCK < ch_per_g:
        BLOCK *= 2
    out = torch.empty_like(x)
    grid = (B * num_groups,)
    gn_leaky_double_simple_kernel[grid](
        x, gamma, beta, out,
        B, C, num_groups, ch_per_g,
        eps, negative_slope,
        BLOCK=BLOCK,
        num_warps=1,
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
        x = triton_linear_tf32(x, self.fc.weight, self.fc.bias)
        x = gn_leaky_double_simple(
            x,
            self.gn.weight.contiguous(),
            self.gn.bias.contiguous(),
            self.num_groups,
            self.eps,
            self.negative_slope,
        )
        return x