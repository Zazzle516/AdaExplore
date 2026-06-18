import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_all_kernel(
    x_ptr,           # [N, C, S]  conv output
    mult_ptr,        # [C]
    out_ptr,         # [N, S]
    N, C, S,
    clamp_min,
    clamp_max,
    eps,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # One program per (n, s_block). Loads full [C, BLOCK_S] tile,
    # computes per-channel mean/var across full S in one pass using a loop,
    # then normalizes the local tile and max-reduces across C.
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    m = tl.load(mult_ptr + c_offs, mask=c_mask, other=0.0)  # [BLOCK_C]

    # Compute per-channel mean & invstd by looping over S in BLOCK_S chunks.
    sum_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sumsq_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    base = pid_n * C * S
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        ptrs = x_ptr + base + c_offs[:, None] * S + offs[None, :]
        full_mask = c_mask[:, None] & mask[None, :]
        x = tl.load(ptrs, mask=full_mask, other=0.0)
        y = x * m[:, None]
        y = tl.where(full_mask, y, 0.0)
        sum_acc += tl.sum(y, axis=1)
        sumsq_acc += tl.sum(y * y, axis=1)

    inv_S = 1.0 / S
    mean = sum_acc * inv_S
    var = sumsq_acc * inv_S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    # Now load the tile for this program's s-block and produce output.
    x_ptrs = x_ptr + base + c_offs[:, None] * S + s_offs[None, :]
    full_mask = c_mask[:, None] & s_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0)

    y = x * m[:, None]
    y = (y - mean[:, None]) * invstd[:, None]
    y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
    y = y * m[:, None]

    neg_inf = float("-inf")
    y_masked = tl.where(c_mask[:, None], y, neg_inf)
    out = tl.max(y_masked, axis=0)

    tl.store(out_ptr + pid_n * S + s_offs, out, mask=s_mask)


def fused_post_conv(x, multiplier, clamp_min, clamp_max, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    x_flat = x.contiguous().view(N, C, S)
    mult_flat = multiplier.contiguous().view(C)

    out = torch.empty((N, S), device=x.device, dtype=torch.float32)

    BLOCK_S = 512
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = (N, (S + BLOCK_S - 1) // BLOCK_S)
    fused_all_kernel[grid](
        x_flat, mult_flat, out,
        N, C, S,
        clamp_min, clamp_max, eps,
        BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=2,
    )
    return out.view(N, D, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        return fused_post_conv(x, self.multiplier, self.clamp_min, self.clamp_max, eps=1e-5)