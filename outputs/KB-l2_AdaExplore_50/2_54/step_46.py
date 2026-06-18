import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_act_kernel_nhwc(
    x_ptr, m_ptr, out_ptr,
    N_elem, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_elem

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # channels_last layout: channel is innermost, so c = offs % C
    c_idx = offs % C
    m = tl.load(m_ptr + c_idx, mask=mask, other=0.0)

    v = x * m
    # LeakyReLU negative_slope=0.01
    v = tl.where(v >= 0, v, v * 0.01)
    # exact GELU
    inv_sqrt2 = 0.7071067811865475
    v = 0.5 * v * (1.0 + tl.erf(v * inv_sqrt2))

    tl.store(out_ptr + offs, v, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv = self.conv.to(memory_format=torch.channels_last)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # channels_last layout enables faster NHWC cuDNN kernels on Ada/Ampere.
        x = x.to(memory_format=torch.channels_last)
        y = self.conv(x)  # NHWC-contiguous output

        N, C, H, W = y.shape
        N_elem = y.numel()

        m = self.multiplier.contiguous().view(-1)
        out = torch.empty_like(y, memory_format=torch.channels_last)

        BLOCK_SIZE = 4096
        grid = (triton.cdiv(N_elem, BLOCK_SIZE),)
        fused_act_kernel_nhwc[grid](
            y, m, out,
            N_elem, C,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8, num_stages=2,
        )
        return out