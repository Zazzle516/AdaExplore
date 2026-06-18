import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_kernel(
    x_ptr,        # [N, D, H, W, C] channels_last
    out_ptr,      # [N, 1, D, H, W]
    bias_ptr,     # scalar
    N, C, D, H, W,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * D * H * W
    if pid >= total:
        return

    base = pid * C  # channels_last: C is innermost contiguous

    c_offs = tl.arange(0, BLOCK_C)
    mask = c_offs < C
    ptrs = x_ptr + base + c_offs
    vals = tl.load(ptrs, mask=mask, other=-float('inf'))

    m = tl.max(vals, axis=0)
    e = tl.exp(vals - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    lse = m + tl.log(s)

    # hardswish: x * sigmoid(x+3) / 6
    sig = 1.0 / (1.0 + tl.exp(-(lse + 3.0)))
    hs = lse * sig / 6.0

    b = tl.load(bias_ptr)
    y = hs - b
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    tl.store(out_ptr + pid, y)


def fused_post(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # x is in channels_last_3d format; create a contiguous NDHWC view
    N, C, D, H, W = x.shape
    # permute to NDHWC and ensure contiguous
    x_nhwc = x.permute(0, 2, 3, 4, 1).contiguous()
    out = torch.empty((N, 1, D, H, W), device=x.device, dtype=x.dtype)
    total = N * D * H * W
    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C < 16:
        BLOCK_C = 16
    grid = (total,)
    fused_post_kernel[grid](
        x_nhwc, out, bias.reshape(-1),
        N, C, D, H, W,
        BLOCK_C=BLOCK_C,
        num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Convert weight to channels_last_3d for faster conv on Ampere
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last_3d)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)
        x = fused_post(x, self.bias)
        return x