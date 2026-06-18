import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_tanh_scale_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, hw)
    pid = tl.program_id(0)
    n = pid // HW
    hw = pid % HW

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    base = n * C * HW + offs * HW + hw
    x = tl.load(x_ptr + base, mask=mask, other=-float('inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    y = tl.extra.cuda.libdevice.tanh(sm) * scaling_factor

    tl.store(out_ptr + base, y, mask=mask)


def fused_softmax_tanh_scale(x: torch.Tensor, scaling_factor: float):
    # x: (B, C, 1, H, W) -> treat as (B, C, H*W)
    B, C, D, H, W = x.shape
    assert D == 1
    x_view = x.view(B, C, H * W).contiguous()
    out = torch.empty_like(x_view)
    HW = H * W
    BLOCK_C = triton.next_power_of_2(C)
    grid = (B * HW,)
    softmax_tanh_scale_kernel[grid](
        x_view, out, B, C, HW, scaling_factor,
        BLOCK_C=BLOCK_C, num_warps=2,
    )
    return out.view(B, C, 1, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.mean(dim=2, keepdim=True)
        x = x + self.bias
        x = fused_softmax_tanh_scale(x, self.scaling_factor)
        return x