import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        mask_k = offs_k < (K - k)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def gn_lrelu_kernel(
    X_ptr, Gamma_ptr, Beta_ptr, Out_ptr,
    N, C, G, CG,
    eps, neg_slope,
    GROUPS_PER_PROG: tl.constexpr,
    BLOCK_CG: tl.constexpr,
):
    # one program per (sample, group_block of GROUPS_PER_PROG groups)
    pid = tl.program_id(0)
    groups_per_row = G // GROUPS_PER_PROG
    n = pid // groups_per_row
    gb = pid % groups_per_row
    g_start = gb * GROUPS_PER_PROG

    # 2D tile: [GROUPS_PER_PROG, BLOCK_CG]
    g_offs = tl.arange(0, GROUPS_PER_PROG)
    c_offs = tl.arange(0, BLOCK_CG)
    mask_c = c_offs < CG

    base = n * C + (g_start + g_offs)[:, None] * CG + c_offs[None, :]
    mask = mask_c[None, :]

    x = tl.load(X_ptr + base, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(x, axis=1)   # [GROUPS_PER_PROG]
    sum_x2 = tl.sum(x * x, axis=1)

    mean = sum_x / CG
    var = sum_x2 / CG - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma_offs = (g_start + g_offs)[:, None] * CG + c_offs[None, :]
    gamma = tl.load(Gamma_ptr + gamma_offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + gamma_offs, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean[:, None]) * rstd[:, None] * gamma + beta
    y = tl.where(y >= 0, y, y * neg_slope)
    y = y + y

    tl.store(Out_ptr + base, y, mask=mask)


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # weight is (N, K), we need it as (K, N) effectively. Use strides.
    # B should be (K, N). weight.t() gives strides (1, K) which works.
    wt = weight.t()  # (K, N)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_kernel[grid](
        x, wt, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        wt.stride(0), wt.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def triton_gn_lrelu_double(x, gamma, beta, num_groups, eps, neg_slope):
    N, C = x.shape
    G = num_groups
    CG = C // G
    out = torch.empty_like(x)
    BLOCK_CG = triton.next_power_of_2(CG)
    # pick groups per program to fill warps
    if CG <= 16:
        GROUPS_PER_PROG = 8
    elif CG <= 32:
        GROUPS_PER_PROG = 4
    elif CG <= 64:
        GROUPS_PER_PROG = 2
    else:
        GROUPS_PER_PROG = 1
    while G % GROUPS_PER_PROG != 0:
        GROUPS_PER_PROG //= 2
    grid = (N * (G // GROUPS_PER_PROG),)
    gn_lrelu_kernel[grid](
        x, gamma, beta, out,
        N, C, G, CG,
        eps, neg_slope,
        GROUPS_PER_PROG=GROUPS_PER_PROG,
        BLOCK_CG=BLOCK_CG,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size).cuda()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps).cuda()
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.fc.weight.contiguous()
        b = self.fc.bias.contiguous()
        y = triton_linear(x, w, b)
        out = triton_gn_lrelu_double(y, self.gn.weight.contiguous(), self.gn.bias.contiguous(),
                                      self.num_groups, self.eps, self.negative_slope)
        return out