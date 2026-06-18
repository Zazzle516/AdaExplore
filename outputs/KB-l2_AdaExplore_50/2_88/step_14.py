import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================
# GEMM: y = x @ W^T + b   where x:(M,K), W:(N,K), out:(M,N)
# We store W transposed at init as W_t: (K, N) contiguous so kernel does A@B.
# ============================================================
GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
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
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remain = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remain), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remain) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :].to(tl.float32)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def triton_linear(x, w_t, bias):
    M, K = x.shape
    K2, N = w_t.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_bias_kernel[grid](
        x, w_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


# ============================================================
# Fused GroupNorm + Swish + Mul + Swish
# One program per (sample, group). CPG = out_features / num_groups.
# Here: out_features=8192, num_groups=256 => CPG=32
# ============================================================
@triton.jit
def fused_gn_swish_mul_swish_kernel(
    X_ptr, Y_ptr,
    Gamma_ptr, Beta_ptr, Mul_ptr,
    M, C, G, CPG,
    eps,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs = tl.arange(0, BLOCK)
    mask = offs < CPG

    c_start = pid_g * CPG
    ptrs = X_ptr + pid_m * C + c_start + offs

    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    # mean / var
    n = CPG
    mean = tl.sum(x, axis=0) / n
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)

    # affine
    gamma = tl.load(Gamma_ptr + c_start + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + c_start + offs, mask=mask, other=0.0).to(tl.float32)
    mw = tl.load(Mul_ptr + c_start + offs, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * gamma + beta
    # swish
    y = y * tl.sigmoid(y)
    # mul
    y = y * mw
    # swish
    y = y * tl.sigmoid(y)

    out_ptrs = Y_ptr + pid_m * C + c_start + offs
    tl.store(out_ptrs, y, mask=mask)


def fused_gn_swish_mul_swish(x, gamma, beta, mul_w, num_groups, eps):
    M, C = x.shape
    G = num_groups
    CPG = C // G
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(CPG)
    if BLOCK < 32:
        BLOCK = 32
    grid = (M, G)
    fused_gn_swish_mul_swish_kernel[grid](
        x, out, gamma, beta, mul_w,
        M, C, G, CPG, eps,
        BLOCK=BLOCK,
        num_warps=1,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super().__init__()
        # Re-use nn.Linear/nn.GroupNorm to get the same param init distribution.
        gemm = nn.Linear(in_features, out_features)
        gn = nn.GroupNorm(num_groups, out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.eps = gn.eps

        # Store W transposed so kernel does A @ B with B = W^T : (K, N) contiguous.
        self.register_buffer('weight_t', gemm.weight.detach().t().contiguous())
        self.bias = nn.Parameter(gemm.bias.detach().clone())
        self.gn_weight = nn.Parameter(gn.weight.detach().clone())
        self.gn_bias = nn.Parameter(gn.bias.detach().clone())

        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        x = x.cuda() if not x.is_cuda else x
        x = x.contiguous()
        # GEMM + bias
        y = triton_linear(x, self.weight_t, self.bias)
        # Fused GN + Swish + Mul + Swish
        out = fused_gn_swish_mul_swish(
            y, self.gn_weight, self.gn_bias, self.multiply_weight,
            self.num_groups, self.eps,
        )
        return out