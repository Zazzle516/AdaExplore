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

    # x layout: [N, C, S] contiguous
    x_offset = n * C * S + offs_c * S + s
    x = tl.load(x_ptr + x_offset, mask=mask_c, other=-float('inf'))

    # softmax over C
    x_max = tl.max(x, axis=0)
    x_shift = x - x_max
    x_exp = tl.exp(x_shift)
    # zero out masked
    x_exp = tl.where(mask_c, x_exp, 0.0)
    denom = tl.sum(x_exp, axis=0)
    sm = x_exp / denom

    # subtract per-channel param
    sub = tl.load(sub_ptr + offs_c, mask=mask_c, other=0.0)
    y = sm - sub

    # swish: sigmoid(y) * y
    sig = 1.0 / (1.0 + tl.exp(-y))
    sw = sig * y

    # mask invalid channels with -inf for max
    sw_masked = tl.where(mask_c, sw, -float('inf'))
    res = tl.max(sw_masked, axis=0)

    out_offset = n * S + s
    tl.store(out_ptr + out_offset, res)


def fused_softmax_sub_swish_max(x: torch.Tensor, sub: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and sub.is_cuda
    x = x.contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W

    # reshape view-only
    x_flat = x.view(N, C, S)
    out = torch.empty((N, S), device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C < 16:
        BLOCK_C = 16

    grid = (N * S,)
    fused_softmax_sub_swish_max_kernel[grid](
        x_flat, sub, out,
        N, C, S,
        BLOCK_C=BLOCK_C,
        num_warps=2,
    )
    return out.view(N, D, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding,
                 pool_kernel_size, pool_stride, pool_padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.max_pool = nn.MaxPool3d(
            kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding
        )
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = fused_softmax_sub_swish_max(x, self.subtract)
        return x