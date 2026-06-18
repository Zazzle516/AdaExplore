import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_kernel(
    x_ptr,         # (B, C, H, W) - mean-pooled conv output
    bias_ptr,      # (C,)
    out_ptr,       # (B, C, H, W)
    B, C, H, W,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    HW = H * W
    b = pid // HW
    rem = pid % HW
    h = rem // W
    w = rem % W

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = b * C * HW + h * W + w
    ptrs = x_ptr + base + offs_c * HW

    x = tl.load(ptrs, mask=mask_c, other=-float('inf'))
    bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    x = x + bias

    x_max = tl.max(x, axis=0)
    x_shift = x - x_max
    e = tl.exp(x_shift)
    e = tl.where(mask_c, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    two_x = 2.0 * sm
    e2 = tl.exp(two_x)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = th * scaling_factor

    out_ptrs = out_ptr + base + offs_c * HW
    tl.store(out_ptrs, out, mask=mask_c)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)            # (B, C, D, H, W)
        x = x.mean(dim=2, keepdim=False)      # (B, C, H, W)
        x = x.contiguous()

        B, C, H, W = x.shape
        out = torch.empty_like(x)

        BLOCK_C = 1
        while BLOCK_C < C:
            BLOCK_C *= 2

        bias_flat = self.bias.view(-1).contiguous()

        grid = (B * H * W,)
        fused_post_kernel[grid](
            x, bias_flat, out,
            B, C, H, W,
            float(self.scaling_factor),
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out.unsqueeze(2)