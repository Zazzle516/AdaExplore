import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
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
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear(x, weight_t, bias):
    # x: [M, K], weight_t: [K, N] (pre-transposed), bias: [N]
    M, K = x.shape
    K2, N = weight_t.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
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
    C, CH_PER_G,
    eps,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_start = pid_g * CH_PER_G
    offs = tl.arange(0, BLOCK)
    mask = offs < CH_PER_G

    chan_off = group_start + offs
    row_base = pid_n * C

    x = tl.load(x_ptr + row_base + chan_off, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(x, axis=0)
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


@triton.jit
def fused_gn_swish_mul_swish_kernel_multi(
    x_ptr, gamma_ptr, beta_ptr, mw_ptr, out_ptr,
    C,
    eps,
    CH_PER_G: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Each program handles GROUPS_PER_BLOCK groups for one batch row.
    # BLOCK_C = GROUPS_PER_BLOCK * CH_PER_G (must be power of 2 friendly)
    pid_n = tl.program_id(0)
    pid_gb = tl.program_id(1)

    g_start = pid_gb * GROUPS_PER_BLOCK
    # offsets [GROUPS_PER_BLOCK, CH_PER_G]
    g_off = tl.arange(0, GROUPS_PER_BLOCK)[:, None]
    c_off = tl.arange(0, CH_PER_G)[None, :]
    chan_off = (g_start + g_off) * CH_PER_G + c_off  # [GP, CH_PER_G]

    row_base = pid_n * C

    x = tl.load(x_ptr + row_base + chan_off).to(tl.float32)

    # reduce along channel dim (axis=1)
    mean = tl.sum(x, axis=1) / CH_PER_G  # [GP]
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / CH_PER_G  # [GP]
    rstd = 1.0 / tl.sqrt(var + eps)  # [GP]

    gamma = tl.load(gamma_ptr + chan_off).to(tl.float32)
    beta = tl.load(beta_ptr + chan_off).to(tl.float32)
    mw = tl.load(mw_ptr + chan_off).to(tl.float32)

    y = xc * rstd[:, None] * gamma + beta
    s1 = y * tl.sigmoid(y)
    z = s1 * mw
    out = z * tl.sigmoid(z)

    tl.store(out_ptr + row_base + chan_off, out)


def fused_gn_swish_mul_swish(x, gamma, beta, mw, num_groups, eps=1e-5):
    N, C = x.shape
    assert C % num_groups == 0
    CH_PER_G = C // num_groups
    out = torch.empty_like(x)

    # Pack multiple groups per program if CH_PER_G is small (e.g., 32).
    # Target BLOCK_C around 256 for good occupancy.
    if CH_PER_G <= 32 and (num_groups % 16 == 0):
        GROUPS_PER_BLOCK = 16
        BLOCK_C = GROUPS_PER_BLOCK * CH_PER_G
        grid = (N, num_groups // GROUPS_PER_BLOCK)
        fused_gn_swish_mul_swish_kernel_multi[grid](
            x, gamma, beta, mw, out,
            C, eps,
            CH_PER_G=CH_PER_G,
            GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
            BLOCK_C=BLOCK_C,
            num_warps=8,
            num_stages=2,
        )
    elif CH_PER_G <= 64 and (num_groups % 8 == 0):
        GROUPS_PER_BLOCK = 8
        BLOCK_C = GROUPS_PER_BLOCK * CH_PER_G
        grid = (N, num_groups // GROUPS_PER_BLOCK)
        fused_gn_swish_mul_swish_kernel_multi[grid](
            x, gamma, beta, mw, out,
            C, eps,
            CH_PER_G=CH_PER_G,
            GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )
    else:
        BLOCK = triton.next_power_of_2(CH_PER_G)
        grid = (N, num_groups)
        fused_gn_swish_mul_swish_kernel[grid](
            x, gamma, beta, mw, out,
            C, CH_PER_G, eps,
            BLOCK=BLOCK,
            num_warps=4 if BLOCK <= 128 else 8,
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
        self._weight_t_cache = None

    def _get_weight_t(self):
        w = self.gemm.weight
        # Cache the transposed (and contiguous) weight for K-contiguous access
        if (self._weight_t_cache is None
                or self._weight_t_cache.data_ptr() == 0
                or self._weight_t_cache.shape[0] != w.shape[1]
                or self._weight_t_cache.device != w.device
                or self._weight_t_cache.dtype != w.dtype):
            self._weight_t_cache = w.t().contiguous()
        return self._weight_t_cache

    def forward(self, x):
        x = x.contiguous()
        if x.is_cuda and x.dtype == torch.float32:
            weight_t = self._get_weight_t()
            x = triton_linear(x, weight_t, self.gemm.bias)
        else:
            x = self.gemm(x)
            x = x.contiguous()
        out = fused_gn_swish_mul_swish(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.multiply_weight,
            self.num_groups,
            self.eps,
        )
        return out