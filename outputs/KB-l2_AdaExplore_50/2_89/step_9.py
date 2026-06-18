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
    # one program per (n, s) pair
    pid = tl.program_id(0)
    n = pid // S
    s = pid % S

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = n * C * S + s
    x_ptrs = x_ptr + base + offs_c * S

    x = tl.load(x_ptrs, mask=mask_c, other=-float('inf'))

    # softmax
    x_max = tl.max(x, axis=0)
    x_shift = x - x_max
    e = tl.exp(x_shift)
    e = tl.where(mask_c, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    # subtract
    sub = tl.load(sub_ptr + offs_c, mask=mask_c, other=0.0)
    y = sm - sub

    # swish: sigmoid(y) * y
    sig = 1.0 / (1.0 + tl.exp(-y))
    sw = sig * y

    # mask out invalid channels with -inf for max reduction
    sw = tl.where(mask_c, sw, -float('inf'))
    res = tl.max(sw, axis=0)

    tl.store(out_ptr + n * S + s, res)


def fused_post(x, sub):
    # x: [N, C, D, H, W] contiguous
    N, C, D, H, W = x.shape
    S = D * H * W
    x_flat = x.contiguous().view(N, C, S)
    out = torch.empty((N, S), device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C < 16:
        BLOCK_C = 16

    grid = (N * S,)
    fused_softmax_sub_swish_max_kernel[grid](
        x_flat, sub, out,
        N, C, S,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out.view(N, D, H, W)


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