import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
    ],
    key=['C', 'S'],
)
@triton.jit
def fused_softmax_sigmoid_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # Each program handles one batch n and BLOCK_S spatial positions.
    # Loads are coalesced along the contiguous spatial axis.
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_start = pid_s * BLOCK_S
    offs_s = s_start + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = pid_n * C * S
    # ptrs shape: [BLOCK_C, BLOCK_S]
    ptrs = x_ptr + base + offs_c[:, None] * S + offs_s[None, :]
    mask = mask_c[:, None] & mask_s[None, :]

    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)               # [BLOCK_S]
    e = tl.exp(x - m[None, :])
    # zero out invalid C lanes for sum
    e = tl.where(mask_c[:, None], e, 0.0)
    s_sum = tl.sum(e, axis=0)           # [BLOCK_S]
    sm = e / s_sum[None, :]
    sig = 1.0 / (1.0 + tl.exp(-sm))
    tl.store(out_ptr + base + offs_c[:, None] * S + offs_s[None, :], sig, mask=mask)


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

    grid = lambda META: (N, triton.cdiv(spatial, META['BLOCK_S']))
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