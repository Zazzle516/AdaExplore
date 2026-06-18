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
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW

    # x layout: (B, C, HW). offset = b*C*HW + c*HW + hw
    x_off = pid_b * C * HW + offs_c[:, None] * HW + offs_hw[None, :]
    mask = mask_c[:, None] & mask_hw[None, :]
    x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    v = x + b[:, None]

    v_safe = tl.where(mask_c[:, None], v, -float('inf'))
    m = tl.max(v_safe, axis=0)  # (BLOCK_HW,)
    e = tl.exp(v - m[None, :])
    e = tl.where(mask_c[:, None], e, 0.0)
    s = tl.sum(e, axis=0)  # (BLOCK_HW,)
    sm = e / s[None, :]

    # tanh via sigmoid: tanh(x) = 2*sigmoid(2x) - 1
    t = 2.0 * tl.sigmoid(2.0 * sm) - 1.0
    out = t * scaling_factor

    tl.store(out_ptr + x_off, out, mask=mask)


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
        BLOCK_HW = 8
        bias_flat = self.bias.view(-1).contiguous()

        grid = (B, triton.cdiv(HW, BLOCK_HW))
        fused_post_kernel[grid](
            x, bias_flat, out,
            B, C, HW,
            float(self.scaling_factor),
            BLOCK_C=BLOCK_C,
            BLOCK_HW=BLOCK_HW,
            num_warps=8,
            num_stages=2,
        )
        return out.view(B, C, 1, H, W)