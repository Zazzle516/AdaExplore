import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_gemm_bias(x, weight_t, bias):
    # x: (M, K), weight_t: (K, N), bias: (N,)
    M, K = x.shape
    K2, N = weight_t.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_bias_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def fused_gn_swish_mul_swish_kernel(
    x_ptr, gamma_ptr, beta_ptr, mw_ptr, out_ptr,
    C, CH_PER_G, GROUPS_PER_PROG: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g_block = tl.program_id(1)

    row_base = pid_n * C
    g_start = pid_g_block * GROUPS_PER_PROG

    offs = tl.arange(0, BLOCK)
    mask = offs < CH_PER_G

    for gi in tl.static_range(0, GROUPS_PER_PROG):
        g = g_start + gi
        group_start = g * CH_PER_G
        chan_off = group_start + offs

        x = tl.load(x_ptr + row_base + chan_off, mask=mask, other=0.0).to(tl.float32)

        sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
        mean = sum_x / CH_PER_G
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / CH_PER_G
        rstd = 1.0 / tl.sqrt(var + eps)

        gamma = tl.load(gamma_ptr + chan_off, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + chan_off, mask=mask, other=0.0).to(tl.float32)
        mw = tl.load(mw_ptr + chan_off, mask=mask, other=0.0).to(tl.float32)

        y = xc * rstd * gamma + beta
        s1 = y * tl.sigmoid(y)
        z = s1 * mw
        out = z * tl.sigmoid(z)

        tl.store(out_ptr + row_base + chan_off, out, mask=mask)


def fused_gn_swish_mul_swish(x, gamma, beta, mw, num_groups, eps=1e-5):
    N, C = x.shape
    CH_PER_G = C // num_groups
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(CH_PER_G)

    # Pack multiple groups per program for small groups
    if CH_PER_G <= 32:
        GROUPS_PER_PROG = 8
    elif CH_PER_G <= 64:
        GROUPS_PER_PROG = 4
    elif CH_PER_G <= 128:
        GROUPS_PER_PROG = 2
    else:
        GROUPS_PER_PROG = 1

    while num_groups % GROUPS_PER_PROG != 0:
        GROUPS_PER_PROG //= 2
    if GROUPS_PER_PROG < 1:
        GROUPS_PER_PROG = 1

    num_blocks_g = num_groups // GROUPS_PER_PROG
    grid = (N, num_blocks_g)

    if BLOCK <= 32:
        num_warps = 1
    elif BLOCK <= 128:
        num_warps = 2
    else:
        num_warps = 4

    fused_gn_swish_mul_swish_kernel[grid](
        x, gamma, beta, mw, out,
        C, CH_PER_G, GROUPS_PER_PROG,
        eps,
        BLOCK=BLOCK,
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

        # Pre-transpose weight to (K, N) contiguous for coalesced loads
        with torch.no_grad():
            wt = self.gemm.weight.detach().t().contiguous()
        self.register_buffer('weight_t', wt)

    def forward(self, x):
        x = x.contiguous()
        # Keep weight_t in sync if weight changed (eval mode assumption)
        gemm_out = triton_gemm_bias(x, self.weight_t, self.gemm.bias)
        out = fused_gn_swish_mul_swish(
            gemm_out,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            self.num_groups,
            self.eps,
        )
        return out