import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_sigmoid_kernel_cl(
    x_ptr, out_ptr,
    NUM_POS,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_C
    offs = tl.arange(0, BLOCK_C)
    ptrs = base + offs
    x = tl.load(x_ptr + ptrs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s_sum = tl.sum(e, axis=0)
    inv = 1.0 / s_sum
    sm = e * inv
    out = 1.0 / (1.0 + tl.exp(-sm))
    tl.store(out_ptr + ptrs, out)


def softmax_sigmoid(x):
    N, C, D, H, W = x.shape
    x_cl = x.contiguous(memory_format=torch.channels_last_3d)
    out_cl = torch.empty_like(x_cl)
    NUM_POS = N * D * H * W
    # C=64 is power of 2
    softmax_sigmoid_kernel_cl[(NUM_POS,)](
        x_cl, out_cl, NUM_POS, BLOCK_C=C, num_warps=1, num_stages=2
    )
    return out_cl


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )
        self.conv_transpose.weight.data = self.conv_transpose.weight.data.contiguous(
            memory_format=torch.channels_last_3d
        )

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)
        x = softmax_sigmoid(x)
        return x