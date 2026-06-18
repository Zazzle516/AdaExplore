import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_kernel(
    x_ptr,        # (B, C, H, W) after mean pool over D
    bias_ptr,     # (C,)
    out_ptr,      # (B, C, H, W)
    B, C, HW,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    x_off = pid_b * C * HW + offs_c * HW + pid_hw
    x = tl.load(x_ptr + x_off, mask=mask_c, other=-float('inf'))
    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    v = x + b

    v_safe = tl.where(mask_c, v, -float('inf'))
    m = tl.max(v_safe, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask_c, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # tanh via exp
    t = (tl.exp(sm) - tl.exp(-sm)) / (tl.exp(sm) + tl.exp(-sm))
    out = t * scaling_factor

    tl.store(out_ptr + x_off, out, mask=mask_c)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)  # (B, C, D, H, W)
        x = x.mean(dim=2)           # (B, C, H, W)
        x = x.contiguous()
        B, C, H, W = x.shape
        HW = H * W
        out = torch.empty_like(x)

        BLOCK_C = triton.next_power_of_2(C)
        bias_flat = self.bias.view(-1).contiguous()

        grid = (B, HW)
        fused_post_kernel[grid](
            x, bias_flat, out,
            B, C, HW,
            float(self.scaling_factor),
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out.view(B, C, 1, H, W)