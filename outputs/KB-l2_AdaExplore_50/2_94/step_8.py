import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# ---------------------------------------------------------------------------
# Fused GEMM + bias + bias2 + Hardtanh + Mish kernel
# Computes: Y = mish(hardtanh(X @ W^T + b_linear + b_extra))
# X: (M, K), W: (N, K), b_linear: (N,), b_extra: (N,)
# Output: (M, N)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_act_kernel(
    x_ptr, w_ptr, b_ptr,
    y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
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

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        w = tl.load(w_ptrs, mask=(mask_n[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add fused bias (linear bias + extra bias)
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # Hardtanh: clamp to [-1, 1]
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    # softplus stable: log1p(exp(x)) but for x in [-1,1], exp won't overflow
    sp = tl.log(1.0 + tl.exp(acc))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = acc * th

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# GroupNorm kernel: per (row, group) compute mean/var, normalize, apply gamma/beta
# Input shape: (M, N) where N = num_groups * channels_per_group
# ---------------------------------------------------------------------------

@triton.jit
def group_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    M, N, G, C,  # C = channels per group, N = G*C
    eps,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // G
    grp = pid % G

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    base = row * N + grp * C
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # mean
    s = tl.sum(x, axis=0)
    mean = s / C
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(gamma_ptr + grp * C + offs, mask=mask, other=0.0)
    b = tl.load(beta_ptr + grp * C + offs, mask=mask, other=0.0)

    y = (x - mean) * rstd * g + b
    tl.store(y_ptr + base + offs, y, mask=mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p <<= 1
    return p


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        # Linear weight/bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.linear_bias = nn.Parameter(torch.empty(out_features))
        # Match nn.Linear default init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.linear_bias, -bound, bound)

        # Extra bias
        self.extra_bias = nn.Parameter(torch.randn(bias_shape))

        # GroupNorm params
        self.gn_weight = nn.Parameter(torch.ones(out_features))
        self.gn_bias = nn.Parameter(torch.zeros(out_features))
        self.eps = 1e-5

        assert out_features % num_groups == 0
        self.channels_per_group = out_features // num_groups
        self.block_c = _next_pow2(self.channels_per_group)

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features

        # Fuse linear bias + extra bias for the GEMM epilogue
        fused_bias = self.linear_bias + self.extra_bias

        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        fused_gemm_act_kernel[grid](
            x, self.weight, fused_bias, y,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            y.stride(0), y.stride(1),
        )

        # GroupNorm
        out = torch.empty_like(y)
        G = self.num_groups
        C = self.channels_per_group
        grid_gn = (M * G,)
        group_norm_kernel[grid_gn](
            y, self.gn_weight, self.gn_bias, out,
            M, N, G, C,
            self.eps,
            BLOCK_C=self.block_c,
        )
        return out