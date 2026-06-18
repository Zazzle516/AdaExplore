import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_sigmoid_kernel_contigC(
    x_ptr, out_ptr,
    M, C,
    BLOCK_C: tl.constexpr,
):
    # One program per spatial-batch row; channel axis is contiguous.
    pid = tl.program_id(0)
    base = pid * C
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C
    ptrs = x_ptr + base + offs_c

    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s_sum = tl.sum(e, axis=0)
    sm = e / s_sum
    sig = 1.0 / (1.0 + tl.exp(-sm))
    tl.store(out_ptr + base + offs_c, sig, mask=mask)


@triton.jit
def fused_softmax_sigmoid_kernel_strided(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // S
    s = pid % S

    base = n * C * S + s
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C
    ptrs = x_ptr + base + offs_c * S

    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s_sum = tl.sum(e, axis=0)
    sm = e / s_sum
    sig = 1.0 / (1.0 + tl.exp(-sm))
    tl.store(out_ptr + base + offs_c * S, sig, mask=mask)


def fused_softmax_sigmoid(x):
    N, C = x.shape[0], x.shape[1]
    D, H, W = x.shape[2], x.shape[3], x.shape[4]
    spatial = D * H * W

    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    # Check if channels_last_3d: stride[1] == 1
    if x.stride(1) == 1 and x.is_contiguous(memory_format=torch.channels_last_3d):
        out = torch.empty_like(x, memory_format=torch.channels_last_3d)
        M = N * spatial
        grid = (M,)
        fused_softmax_sigmoid_kernel_contigC[grid](
            x, out,
            M, C,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )
        return out
    else:
        x_c = x.contiguous()
        out = torch.empty_like(x_c)
        grid = (N * spatial,)
        fused_softmax_sigmoid_kernel_strided[grid](
            x_c, out,
            N, C, spatial,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )
        return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )
        # Use channels_last_3d layout for better memory access in softmax
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)
        x = fused_softmax_sigmoid(x)
        return x