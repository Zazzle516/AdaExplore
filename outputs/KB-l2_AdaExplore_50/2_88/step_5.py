import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


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
    if CH_PER_G <= 64 and (num_groups % 8 == 0):
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

    def forward(self, x):
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