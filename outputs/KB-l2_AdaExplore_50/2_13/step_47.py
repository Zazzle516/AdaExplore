import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_mean_bias_softmax_tanh_scale_kernel(
    x_ptr, bias_ptr, out_ptr,
    B, C, D, H, W,
    scaling_factor,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # one program per (n, h, w)
    pid = tl.program_id(0)
    HW = H * W
    n = pid // HW
    hw = pid % HW
    h = hw // W
    w = hw % W

    offs_c = tl.arange(0, BLOCK_C)
    offs_d = tl.arange(0, BLOCK_D)
    mask_c = offs_c < C
    mask_d = offs_d < D

    # x stride: (C*D*H*W, D*H*W, H*W, W, 1)
    # load x[n, c, d, h, w] for c in [0,C), d in [0,D)
    base = n * C * D * HW + h * W + w
    ptrs = base + offs_c[:, None] * D * HW + offs_d[None, :] * HW
    mask = mask_c[:, None] & mask_d[None, :]
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0)

    # mean over D
    s_d = tl.sum(x, axis=1)
    mean = s_d / D  # shape [BLOCK_C]

    # add bias
    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    v = mean + b
    v = tl.where(mask_c, v, -float('inf'))

    # softmax over C
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask_c, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    y = tl.extra.cuda.libdevice.tanh(sm) * scaling_factor

    # store to (B, C, 1, H, W) -> stride (C*HW, HW, HW, W, 1)
    out_base = n * C * HW + h * W + w
    out_ptrs = out_base + offs_c * HW
    tl.store(out_ptr + out_ptrs, y, mask=mask_c)


def fused_mean_bias_softmax_tanh_scale(x: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    # x: (B, C, D, H, W)
    B, C, D, H, W = x.shape
    out = torch.empty((B, C, 1, H, W), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_D = triton.next_power_of_2(D)
    grid = (B * H * W,)
    fused_mean_bias_softmax_tanh_scale_kernel[grid](
        x, bias, out, B, C, D, H, W, scaling_factor,
        BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D, num_warps=2,
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
        bias_flat = self.bias.view(-1).contiguous()
        x = fused_mean_bias_softmax_tanh_scale(x, bias_flat, self.scaling_factor)
        return x