import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mean_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    B, C, D, HW,
    BLOCK_HW: tl.constexpr,
):
    # one program per (b, c, hw_block)
    pid = tl.program_id(0)
    hw_block = tl.program_id(1)
    b = pid // C
    c = pid % C

    offs = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < HW

    base = b * C * D * HW + c * D * HW
    inv_d = 1.0 / D

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    for d in range(D):
        ptrs = x_ptr + base + d * HW + offs
        v = tl.load(ptrs, mask=mask, other=0.0)
        acc += v
    acc = acc * inv_d
    bias = tl.load(bias_ptr + c)
    acc += bias

    out_ptrs = out_ptr + b * C * HW + c * HW + offs
    tl.store(out_ptrs, acc, mask=mask)


@triton.jit
def softmax_tanh_scale_kernel(
    x_ptr, out_ptr,
    B, C, HW,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    # one program per (b, hw_idx)
    pid = tl.program_id(0)
    b = pid // HW
    hw = pid % HW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # input layout: (B, C, 1, H, W) contiguous -> stride = (C*HW, HW, HW, W, 1)
    # index x[b, c, 0, h, w] = b*C*HW + c*HW + hw
    base = b * C * HW + hw
    ptrs = x_ptr + base + offs_c * HW

    x = tl.load(ptrs, mask=mask_c, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask_c, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    # tanh via exp
    # tanh(y) = (exp(2y)-1)/(exp(2y)+1)
    e2 = tl.exp(2.0 * sm)
    t = (e2 - 1.0) / (e2 + 1.0)
    out = t * scaling_factor

    tl.store(out_ptr + base + offs_c * HW, out, mask=mask_c)


def fused_softmax_tanh_scale(x, scaling_factor):
    # x shape: (B, C, 1, H, W)
    B, C, D, H, W = x.shape
    assert D == 1
    HW = H * W
    x = x.contiguous()
    out = torch.empty_like(x)

    BLOCK_C = triton.next_power_of_2(C)
    grid = (B * HW,)
    softmax_tanh_scale_kernel[grid](
        x, out,
        B, C, HW,
        scaling_factor,
        BLOCK_C=BLOCK_C,
        num_warps=1,
    )
    return out


def fused_mean_bias(x, bias):
    # x shape: (B, C, D, H, W) -> (B, C, 1, H, W)
    B, C, D, H, W = x.shape
    HW = H * W
    x = x.contiguous()
    out = torch.empty((B, C, 1, H, W), device=x.device, dtype=x.dtype)
    bias_flat = bias.view(-1).contiguous()

    BLOCK_HW = 1024
    grid = (B * C, triton.cdiv(HW, BLOCK_HW))
    mean_bias_kernel[grid](
        x, bias_flat, out,
        B, C, D, HW,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_mean_bias(x, self.bias)
        x = fused_softmax_tanh_scale(x, self.scaling_factor)
        return x