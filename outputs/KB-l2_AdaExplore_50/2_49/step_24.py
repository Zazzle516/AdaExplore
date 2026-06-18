import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_sigmoid_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // S
    s = pid % S
    if n >= N:
        return
    base = n * C * S + s
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    ptrs = base + offs * S
    x = tl.load(x_ptr + ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s_sum = tl.sum(e, axis=0)
    sm = e / s_sum
    out = 1.0 / (1.0 + tl.exp(-sm))
    tl.store(out_ptr + ptrs, out, mask=mask)


def softmax_sigmoid(x):
    N, C, D, H, W = x.shape
    S = D * H * W
    x = x.contiguous()
    out = torch.empty_like(x)
    # Find next power of 2 >= C
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2
    grid = (N * S,)
    softmax_sigmoid_kernel[grid](x, out, N, C, S, BLOCK_C=BLOCK_C, num_warps=4)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        x = softmax_sigmoid(x)
        return x