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
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    for r in tl.static_range(ROWS_PER_PROG):
        row = row_start + r
        row_mask = row < NUM_POS
        ptrs = row * C + offs_c
        m_load = mask_c & row_mask
        x = tl.load(x_ptr + ptrs, mask=m_load, other=-float('inf'))
        m = tl.max(x, axis=0)
        e = tl.exp(x - m)
        e = tl.where(mask_c, e, 0.0)
        s_sum = tl.sum(e, axis=0)
        inv = 1.0 / s_sum
        sm = e * inv
        out = 1.0 / (1.0 + tl.exp(-sm))
        tl.store(out_ptr + ptrs, out, mask=m_load)


def softmax_sigmoid(x):
    N, C, D, H, W = x.shape
    x_cl = x.contiguous(memory_format=torch.channels_last_3d)
    out_cl = torch.empty_like(x_cl)
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2
    NUM_POS = N * D * H * W
    ROWS_PER_PROG = 4
    grid = ((NUM_POS + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)
    softmax_sigmoid_kernel_cl[grid](
        x_cl, out_cl, NUM_POS, C,
        BLOCK_C=BLOCK_C, ROWS_PER_PROG=ROWS_PER_PROG,
        num_warps=1, num_stages=2
    )
    return out_cl


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )
        # Convert weights to channels_last_3d to encourage cuDNN to pick a CL kernel
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)
        x = softmax_sigmoid(x)
        return x