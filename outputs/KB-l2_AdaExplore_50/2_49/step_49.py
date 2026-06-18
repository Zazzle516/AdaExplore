import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_sigmoid_kernel_cl(
    x_ptr, out_ptr,
    NUM_POS, C,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * C
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    ptrs = base + offs
    x = tl.load(x_ptr + ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s_sum = tl.sum(e, axis=0)
    inv = 1.0 / s_sum
    sm = e * inv
    out = 1.0 / (1.0 + tl.exp(-sm))
    tl.store(out_ptr + ptrs, out, mask=mask)


@triton.jit
def softmax_sigmoid_kernel_cl_block(
    x_ptr, out_ptr,
    NUM_POS, C,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    p_offs = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < NUM_POS
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    # Load tile [BLOCK_P, BLOCK_C]
    ptrs = p_offs[:, None] * C + c_offs[None, :]
    mask = p_mask[:, None] & c_mask[None, :]
    x = tl.load(x_ptr + ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=1)
    e = tl.exp(x - m[:, None])
    e = tl.where(mask, e, 0.0)
    s_sum = tl.sum(e, axis=1)
    inv = 1.0 / s_sum
    sm = e * inv[:, None]
    out = 1.0 / (1.0 + tl.exp(-sm))
    tl.store(out_ptr + ptrs, out, mask=mask)


def softmax_sigmoid(x):
    N, C, D, H, W = x.shape
    x_cl = x.contiguous(memory_format=torch.channels_last_3d)
    out_cl = torch.empty_like(x_cl)
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2
    NUM_POS = N * D * H * W
    BLOCK_P = 8
    grid = ((NUM_POS + BLOCK_P - 1) // BLOCK_P,)
    softmax_sigmoid_kernel_cl_block[grid](
        x_cl, out_cl, NUM_POS, C,
        BLOCK_P=BLOCK_P, BLOCK_C=BLOCK_C,
        num_warps=4, num_stages=2,
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