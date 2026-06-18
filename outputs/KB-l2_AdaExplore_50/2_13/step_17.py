import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _post_kernel(
    x_ptr, bias_ptr, out_ptr,
    B, C, D, H, W,
    scaling_factor,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    HW = H * W

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    mask2d = mask_c[:, None] & mask_hw[None, :]

    acc = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)
    base = pid_b * C * D * HW + offs_c[:, None] * D * HW + offs_hw[None, :]
    for d in range(0, D):
        x_ptrs = x_ptr + base + d * HW
        x = tl.load(x_ptrs, mask=mask2d, other=0.0).to(tl.float32)
        acc += x

    inv_D = 1.0 / D.to(tl.float32)
    acc = acc * inv_D

    bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)

    v = acc + bias[:, None]
    v = tl.where(mask_c[:, None], v, -float('inf'))

    m = tl.max(v, axis=0)
    e = tl.exp(v - m[None, :])
    e = tl.where(mask_c[:, None], e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s[None, :]

    e2 = tl.exp(2.0 * sm)
    t = (e2 - 1.0) / (e2 + 1.0)
    out = t * scaling_factor

    out_ptrs = out_ptr + pid_b * C * HW + offs_c[:, None] * HW + offs_hw[None, :]
    tl.store(out_ptrs, out, mask=mask2d)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)  # (B, C, D, H, W)
        x = x.contiguous()

        B, C, D, H, W = x.shape
        out = torch.empty((B, C, 1, H, W), dtype=x.dtype, device=x.device)

        bias_flat = self.bias.view(-1).contiguous()

        BLOCK_C = 1
        while BLOCK_C < C:
            BLOCK_C *= 2

        BLOCK_HW = 256
        HW = H * W
        grid = (B, (HW + BLOCK_HW - 1) // BLOCK_HW)
        _post_kernel[grid](
            x, bias_flat, out,
            B, C, D, H, W,
            float(self.scaling_factor),
            BLOCK_C=BLOCK_C,
            BLOCK_HW=BLOCK_HW,
            num_warps=8,
            num_stages=2,
        )

        return out