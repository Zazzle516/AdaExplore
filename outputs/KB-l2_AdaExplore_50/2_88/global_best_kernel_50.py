import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_swish_mul_swish_kernel(
    x_ptr, w_ptr, mw_ptr, out_ptr,
    N, C, G, CH_PER_G,
    eps,
    BLOCK: tl.constexpr,
):
    # one program per (batch, group)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_start = pid_g * CH_PER_G
    offs = tl.arange(0, BLOCK)
    mask = offs < CH_PER_G

    row_ptr = x_ptr + pid_n * C + group_start
    x = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # mean / var over group
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = sum_x / CH_PER_G
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / CH_PER_G
    rstd = 1.0 / tl.sqrt(var + eps)

    # load affine + multiply weight for this group's channels
    chan_off = group_start + offs
    gamma = tl.load(w_ptr + chan_off, mask=mask, other=0.0).to(tl.float32)
    beta_ptr = w_ptr  # placeholder; we'll pass beta separately
    # Actually pass beta via separate pointer below

    # We'll reload via a separate kernel arg. Here just use gamma.
    # normalize
    y = xc * rstd * gamma  # missing beta - handled in caller via second ptr

    # store intermediate? No - we'll do full fusion in a single kernel below.
    tl.store(out_ptr + pid_n * C + chan_off, y, mask=mask)


@triton.jit
def fused_gn_swish_mul_swish_kernel(
    x_ptr, gamma_ptr, beta_ptr, mw_ptr, out_ptr,
    C, G, CH_PER_G,
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

    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = sum_x / CH_PER_G
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / CH_PER_G
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + chan_off, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + chan_off, mask=mask, other=0.0).to(tl.float32)
    mw = tl.load(mw_ptr + chan_off, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * gamma + beta
    # swish
    s1 = y * tl.sigmoid(y)
    # multiply weight
    z = s1 * mw
    # swish again
    out = z * tl.sigmoid(z)

    tl.store(out_ptr + row_base + chan_off, out, mask=mask)


def fused_gn_swish_mul_swish(x, gamma, beta, mw, num_groups, eps=1e-5):
    N, C = x.shape
    assert C % num_groups == 0
    CH_PER_G = C // num_groups
    out = torch.empty_like(x)
    # next power of 2 for BLOCK
    BLOCK = triton.next_power_of_2(CH_PER_G)
    grid = (N, num_groups)
    fused_gn_swish_mul_swish_kernel[grid](
        x, gamma, beta, mw, out,
        C, num_groups, CH_PER_G,
        eps,
        BLOCK=BLOCK,
        num_warps=4 if BLOCK <= 128 else 8,
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