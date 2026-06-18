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

    # Initialize max accumulators per channel to -inf
    NEG_INF = float('-inf')
    max_acc = tl.full([BLOCK_C], NEG_INF, dtype=tl.float32)

    d_base = dp * POOL
    h_base = hp * POOL
    w_base = wp * POOL

    # Iterate over the POOL^3 window
    for di in tl.static_range(0, POOL):
        for hi in tl.static_range(0, POOL):
            for wi in tl.static_range(0, POOL):
                d_idx = d_base + di
                h_idx = h_base + hi
                w_idx = w_base + wi

                # Compute softmax over channels for this spatial location.
                # Load all channels at (n, :, d_idx, h_idx, w_idx).
                base = ((n * C + 0) * D + d_idx) * H * W + h_idx * W + w_idx
                stride_c = D * H * W
                ptrs = x_ptr + base + c_offs * stride_c

                vals = tl.load(ptrs, mask=c_mask, other=NEG_INF)
                # numerically stable softmax
                row_max = tl.max(vals, axis=0)
                exps = tl.exp(vals - row_max)
                exps = tl.where(c_mask, exps, 0.0)
                row_sum = tl.sum(exps, axis=0)
                soft = exps / row_sum

                max_acc = tl.maximum(max_acc, soft)

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

    grid = (N * Dp * Hp * Wp,)
    fused_softmax_pool_kernel[grid](
        x, out,
        N, C, D, H, W,
        Dp, Hp, Wp,
        POOL=POOL,
        BLOCK_C=BLOCK_C,
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