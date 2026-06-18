import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_sub_swish_max_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, s)
    pid = tl.program_id(0)
    n = pid // S
    s = pid % S

    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    base = n * C * S + s
    x = tl.load(x_ptr + base + offs * S, mask=mask, other=-float('inf'))

    # softmax over channels
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    z = tl.sum(e, axis=0)
    sm = e / z

    sub = tl.load(sub_ptr + offs, mask=mask, other=0.0)
    y = sm - sub
    # swish: y * sigmoid(y)
    sw = y * tl.sigmoid(y)
    sw = tl.where(mask, sw, -float('inf'))
    out_val = tl.max(sw, axis=0)

    tl.store(out_ptr + n * S + s, out_val)


def fused_post(x, sub):
    # x: (N, C, D, H, W)
    N, C, D, H, W = x.shape
    S = D * H * W
    x_c = x.contiguous()
    out = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * S,)
    fused_softmax_sub_swish_max_kernel[grid](
        x_c, sub.contiguous(), out,
        N, C, S,
        BLOCK_C=BLOCK_C,
        num_warps=1,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = fused_post(x, self.subtract)
        return x