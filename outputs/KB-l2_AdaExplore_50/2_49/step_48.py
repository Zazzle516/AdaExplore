import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=2),
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
    # one program processes one n and BLOCK_S spatial positions, all C channels
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_c = tl.arange(0, BLOCK_C)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    mask_c = offs_c < C
    mask_s = offs_s < S

    base = pid_n * C * S
    # 2D tile: [BLOCK_C, BLOCK_S]
    ptrs = x_ptr + base + offs_c[:, None] * S + offs_s[None, :]
    mask = mask_c[:, None] & mask_s[None, :]

    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)  # [BLOCK_S]
    e = tl.exp(x - m[None, :])
    s_sum = tl.sum(e, axis=0)  # [BLOCK_S]
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

    grid = lambda meta: (N, triton.cdiv(spatial, meta['BLOCK_S']))
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