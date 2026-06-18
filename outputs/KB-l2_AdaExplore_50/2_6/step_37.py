import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_pool_kernel(
    x_ptr,         # input after conv: (N, C, D, H, W)
    out_ptr,       # output: (N, C, Dp, Hp, Wp)
    N, C, D, H, W,
    Dp, Hp, Wp,
    POOL: tl.constexpr,   # combined pool size (4)
    BLOCK_C: tl.constexpr,
    WIN: tl.constexpr,     # POOL*POOL*POOL
):
    # one program per (n, dp, hp, wp)
    pid = tl.program_id(0)
    wp = pid % Wp
    pid2 = pid // Wp
    hp = pid2 % Hp
    pid3 = pid2 // Hp
    dp = pid3 % Dp
    n = pid3 // Dp

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    NEG_INF = float('-inf')

    d_base = dp * POOL
    h_base = hp * POOL
    w_base = wp * POOL

    stride_c = D * H * W
    HW = H * W

    # Build window offsets [WIN] indexing within (D,H,W)
    w_offs = tl.arange(0, WIN)
    wi = w_offs % POOL
    hi = (w_offs // POOL) % POOL
    di = w_offs // (POOL * POOL)
    spatial_off = (d_base + di) * HW + (h_base + hi) * W + (w_base + wi)  # [WIN]

    n_base = n * C * stride_c

    # ptrs shape [BLOCK_C, WIN]
    ptrs = x_ptr + n_base + c_offs[:, None] * stride_c + spatial_off[None, :]
    vals = tl.load(ptrs, mask=c_mask[:, None], other=NEG_INF)

    # softmax along C axis (axis=0) for each of WIN spatial points
    row_max = tl.max(vals, axis=0)               # [WIN]
    exps = tl.exp(vals - row_max[None, :])       # [BLOCK_C, WIN]
    exps = tl.where(c_mask[:, None], exps, 0.0)
    row_sum = tl.sum(exps, axis=0)               # [WIN]
    soft = exps / row_sum[None, :]               # [BLOCK_C, WIN]

    # max-pool over spatial dim (axis=1)
    max_acc = tl.max(soft, axis=1)               # [BLOCK_C]

    # write output: (n, c, dp, hp, wp)
    out_base = ((n * C + 0) * Dp + dp) * Hp * Wp + hp * Wp + wp
    out_stride_c = Dp * Hp * Wp
    out_ptrs = out_ptr + out_base + c_offs * out_stride_c
    tl.store(out_ptrs, max_acc, mask=c_mask)


def fused_softmax_pool(x: torch.Tensor, pool_kernel_size: int):
    N, C, D, H, W = x.shape
    POOL = pool_kernel_size * pool_kernel_size  # two consecutive pools
    Dp = D // POOL
    Hp = H // POOL
    Wp = W // POOL

    x = x.contiguous()
    out = torch.empty((N, C, Dp, Hp, Wp), device=x.device, dtype=x.dtype)

    # next power of 2 for C
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2
    BLOCK_C = max(BLOCK_C, 16)

    WIN = POOL * POOL * POOL
    grid = (N * Dp * Hp * Wp,)
    fused_softmax_pool_kernel[grid](
        x, out,
        N, C, D, H, W,
        Dp, Hp, Wp,
        POOL=POOL,
        BLOCK_C=BLOCK_C,
        WIN=WIN,
        num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        # Combined pool window = pool_kernel_size * pool_kernel_size in each dim
        # Need spatial sizes to be divisible by POOL^2; if not, fallback.
        POOL = self.pool_kernel_size * self.pool_kernel_size
        N, C, D, H, W = x.shape
        if D % POOL == 0 and H % POOL == 0 and W % POOL == 0:
            return fused_softmax_pool(x, self.pool_kernel_size)
        else:
            x = torch.softmax(x, dim=1)
            x = F.max_pool3d(x, self.pool_kernel_size)
            x = F.max_pool3d(x, self.pool_kernel_size)
            return x