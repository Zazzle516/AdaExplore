import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr,
    sum_ptr,
    out_ptr,
    spatial,
    total_elements,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    nc = offsets // spatial
    c = nc % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    s = tl.load(sum_ptr + c, mask=mask, other=0.0)

    # LeakyReLU(0.2)
    x = tl.where(x > 0, x, x * 0.2)
    # Add per-channel
    x = x + s
    # Clamp [-1, 1]
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    # GELU exact
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offsets, x, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)
        N, C, D, H, W = x.shape
        x = x.contiguous()
        sum_flat = self.sum_tensor.contiguous().view(-1)
        out = torch.empty_like(x)
        total = x.numel()
        spatial = D * H * W
        BLOCK_SIZE = 2048
        grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        fused_epilogue_kernel[grid](
            x, sum_flat, out,
            spatial, total, C,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )
        return out