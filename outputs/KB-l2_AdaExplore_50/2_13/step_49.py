import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_meanD_bias_softmax_tanh_scale_kernel(
    x_ptr, bias_ptr, out_ptr,
    B, C, D, HW,
    scaling_factor,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # one program per (n, hw)
    pid = tl.program_id(0)
    n = pid // HW
    hw = pid % HW

    c_offs = tl.arange(0, BLOCK_C)
    d_offs = tl.arange(0, BLOCK_D)
    c_mask = c_offs < C
    d_mask = d_offs < D

    # x layout: (B, C, D, H*W) contiguous
    # base address for (n, c, d, hw) = n*C*D*HW + c*D*HW + d*HW + hw
    addrs = n * C * D * HW + c_offs[:, None] * D * HW + d_offs[None, :] * HW + hw
    mask2d = c_mask[:, None] & d_mask[None, :]
    vals = tl.load(x_ptr + addrs, mask=mask2d, other=0.0)
    # mean over D
    summed = tl.sum(vals, axis=1)  # (BLOCK_C,)
    mean = summed / D

    # add bias (per channel)
    bias = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0)
    z = mean + bias

    # softmax across C
    z_masked = tl.where(c_mask, z, -float('inf'))
    m = tl.max(z_masked, axis=0)
    e = tl.exp(z_masked - m)
    e = tl.where(c_mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    y = tl.extra.cuda.libdevice.tanh(sm) * scaling_factor

    # output layout: (B, C, 1, H, W) -> flatten as (B, C, HW)
    out_addrs = n * C * HW + c_offs * HW + hw
    tl.store(out_ptr + out_addrs, y, mask=c_mask)


def fused_meanD_bias_softmax_tanh_scale(x: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    # x: (B, C, D, H, W)
    B, C, D, H, W = x.shape
    HW = H * W
    x_c = x.contiguous()
    bias_flat = bias.contiguous().view(-1)
    out = torch.empty((B, C, 1, H, W), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_D = triton.next_power_of_2(D)
    grid = (B * HW,)
    fused_meanD_bias_softmax_tanh_scale_kernel[grid](
        x_c, bias_flat, out,
        B, C, D, HW, scaling_factor,
        BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_meanD_bias_softmax_tanh_scale(x, self.bias, self.scaling_factor)
        return x