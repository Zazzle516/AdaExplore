import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_swish_bias_kernel(
    A_ptr, W_ptr, Lbias_ptr, Bias_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_remaining), other=0.0)
        acc += tl.dot(a, w)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    lbias = tl.load(Lbias_ptr + offs_n, mask=offs_n < N, other=0.0)
    bbias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + lbias[None, :]
    # swish
    acc = acc * tl.sigmoid(acc)
    acc = acc + bbias[None, :]

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask)


@triton.jit
def group_norm_kernel(
    X_ptr, Gamma_ptr, Beta_ptr, Out_ptr,
    M, N, num_groups, group_size, eps,
    BLOCK_GS: tl.constexpr, GROUPS_PER_PROG: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_start = pid_g * GROUPS_PER_PROG
    offs_gs = tl.arange(0, BLOCK_GS)

    for gi in tl.static_range(GROUPS_PER_PROG):
        g = g_start + gi
        # mask group < num_groups handled by program count being exact
        col_start = g * group_size
        offs = col_start + offs_gs
        x_ptrs = X_ptr + pid_m * N + offs
        mask = offs_gs < group_size
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        sum_x = tl.sum(x, axis=0)
        sum_x2 = tl.sum(x * x, axis=0)
        mean = sum_x / group_size
        var = sum_x2 / group_size - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        gamma = tl.load(Gamma_ptr + offs, mask=mask, other=0.0)
        beta = tl.load(Beta_ptr + offs, mask=mask, other=0.0)
        y = (x - mean) * rstd * gamma + beta
        tl.store(Out_ptr + pid_m * N + offs, y, mask=mask)


def triton_gemm_swish_bias(x, weight, lbias, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_swish_bias_kernel[grid](
        x, weight, lbias, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def triton_group_norm(x, gamma, beta, num_groups, eps):
    M, N = x.shape
    group_size = N // num_groups
    BLOCK_GS = triton.next_power_of_2(group_size)
    GROUPS_PER_PROG = 4
    while num_groups % GROUPS_PER_PROG != 0 and GROUPS_PER_PROG > 1:
        GROUPS_PER_PROG //= 2
    out = torch.empty_like(x)
    grid = (M, num_groups // GROUPS_PER_PROG)
    group_norm_kernel[grid](
        x, gamma, beta, out,
        M, N, num_groups, group_size, eps,
        BLOCK_GS=BLOCK_GS, GROUPS_PER_PROG=GROUPS_PER_PROG,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        w = self.matmul.weight.contiguous()
        lbias = self.matmul.bias.contiguous()
        bias = self.bias.contiguous()
        y = triton_gemm_swish_bias(x, w, lbias, bias)
        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        out = triton_group_norm(y, gamma, beta, self.num_groups, self.eps)
        return out