import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gn_swish_mul_swish_kernel(
    x_ptr,          # (M, N)
    gamma_ptr,      # (N,)
    beta_ptr,       # (N,)
    mw_ptr,         # (N,)
    out_ptr,        # (M, N)
    M, N,
    G: tl.constexpr,            # num groups
    GROUP_SIZE: tl.constexpr,   # N // G
    GROUPS_PER_BLOCK: tl.constexpr,
    eps: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)  # which chunk of groups

    g_start = pid_g * GROUPS_PER_BLOCK
    # offsets within the row for this chunk of groups
    # shape: [GROUPS_PER_BLOCK, GROUP_SIZE]
    g_off = tl.arange(0, GROUPS_PER_BLOCK)  # group index within block
    c_off = tl.arange(0, GROUP_SIZE)        # channel within group

    # global channel index: (g_start + g_off) * GROUP_SIZE + c_off
    col_idx = (g_start + g_off)[:, None] * GROUP_SIZE + c_off[None, :]  # [GPB, GS]
    row_off = pid_m * N
    ptrs = x_ptr + row_off + col_idx

    g_mask = (g_start + g_off) < G  # [GPB]
    mask = g_mask[:, None]

    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    # per-group mean/var: reduce along axis=1 (GROUP_SIZE)
    mean = tl.sum(x, axis=1) / GROUP_SIZE  # [GPB]
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, axis=1) / GROUP_SIZE  # [GPB]
    rstd = 1.0 / tl.sqrt(var + eps)  # [GPB]

    x_norm = xc * rstd[:, None]

    # load gamma, beta, mw for these channels
    gamma = tl.load(gamma_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
    mw = tl.load(mw_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)

    y = x_norm * gamma + beta
    # swish
    y = y * tl.sigmoid(y)
    # multiply
    y = y * mw
    # swish
    y = y * tl.sigmoid(y)

    tl.store(out_ptr + row_off + col_idx, y, mask=mask)


def fused_gn_swish_mul_swish(x, gamma, beta, mw, num_groups, eps=1e-5):
    M, N = x.shape
    GROUP_SIZE = N // num_groups
    G = num_groups
    out = torch.empty_like(x)

    # pick GROUPS_PER_BLOCK
    if GROUP_SIZE <= 32:
        GROUPS_PER_BLOCK = 8
    elif GROUP_SIZE <= 64:
        GROUPS_PER_BLOCK = 4
    elif GROUP_SIZE <= 128:
        GROUPS_PER_BLOCK = 2
    else:
        GROUPS_PER_BLOCK = 1

    # ensure GROUPS_PER_BLOCK divides G or we mask
    num_g_blocks = (G + GROUPS_PER_BLOCK - 1) // GROUPS_PER_BLOCK

    grid = (M, num_g_blocks)

    num_warps = 4
    elements_per_block = GROUPS_PER_BLOCK * GROUP_SIZE
    if elements_per_block >= 512:
        num_warps = 8
    if elements_per_block <= 64:
        num_warps = 2

    fused_gn_swish_mul_swish_kernel[grid](
        x, gamma, beta, mw, out,
        M, N,
        G=G,
        GROUP_SIZE=GROUP_SIZE,
        GROUPS_PER_BLOCK=GROUPS_PER_BLOCK,
        eps=eps,
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

    def forward(self, x):
        x = F.linear(x, self.gemm.weight, self.gemm.bias)
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