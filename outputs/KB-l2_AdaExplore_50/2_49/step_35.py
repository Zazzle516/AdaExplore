import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['C', 'S'],
)
@triton.jit
def fused_softmax_sigmoid_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, s) - reduces over C
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
    # x: (N, C, D, H, W)
    N, C = x.shape[0], x.shape[1]
    spatial = x.shape[2] * x.shape[3] * x.shape[4]
    x_c = x.contiguous()
    out = torch.empty_like(x_c)

    # next power of 2 >= C
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = (N * spatial,)
    fused_softmax_sigmoid_kernel[grid](
        x_c, out,
        N, C, spatial,
        BLOCK_C=BLOCK_C,
    )
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
        x = fused_softmax_sigmoid(x)
        return x