import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
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
    N, C, G,
    eps, neg_slope,
    CG: tl.constexpr,
    GROUPS_PER_PROG: tl.constexpr,
):
    # one program processes GROUPS_PER_PROG groups for one sample row
    pid = tl.program_id(0)
    groups_per_row = G // GROUPS_PER_PROG
    n = pid // groups_per_row
    g_block = pid % groups_per_row
    g_start = g_block * GROUPS_PER_PROG

    offs_c = tl.arange(0, CG)
    offs_g = tl.arange(0, GROUPS_PER_PROG)

    # base offset within row for this group block
    chan_base = g_start * CG
    # load [GROUPS_PER_PROG, CG] tile
    row_off = n * C + chan_base + offs_g[:, None] * CG + offs_c[None, :]
    x = tl.load(X_ptr + row_off).to(tl.float32)

    sum_x = tl.sum(x, axis=1)
    sum_x2 = tl.sum(x * x, axis=1)

    mean = sum_x / CG
    var = sum_x2 / CG - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma_off = chan_base + offs_g[:, None] * CG + offs_c[None, :]
    gamma = tl.load(Gamma_ptr + gamma_off).to(tl.float32)
    beta = tl.load(Beta_ptr + gamma_off).to(tl.float32)

    y = (x - mean[:, None]) * rstd[:, None] * gamma + beta
    y = tl.where(y >= 0, y, y * neg_slope)
    y = y + y

    tl.store(Out_ptr + row_off, y)


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
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
    # pick groups per program to amortize launches
    if CG <= 32:
        GROUPS_PER_PROG = 8
    elif CG <= 64:
        GROUPS_PER_PROG = 4
    else:
        GROUPS_PER_PROG = 1
    while G % GROUPS_PER_PROG != 0:
        GROUPS_PER_PROG //= 2
    grid = (N * (G // GROUPS_PER_PROG),)
    gn_lrelu_kernel[grid](
        x, gamma, beta, out,
        N, C, G,
        eps, neg_slope,
        CG=CG,
        GROUPS_PER_PROG=GROUPS_PER_PROG,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope
        # move params to CUDA at construction
        self.cuda()

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.fc.weight.contiguous()
        b = self.fc.bias.contiguous()
        y = triton_linear(x, w, b)
        out = triton_gn_lrelu_double(y, self.gn.weight.contiguous(), self.gn.bias.contiguous(),
                                      self.num_groups, self.eps, self.negative_slope)
        return out