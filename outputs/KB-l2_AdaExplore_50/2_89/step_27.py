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
    BLOCK_S: tl.constexpr,
):
    # one program per (n, s_block) — process BLOCK_S spatial points at once
    pid = tl.program_id(0)
    num_s_blocks = (S + BLOCK_S - 1) // BLOCK_S
    n = pid // num_s_blocks
    sb = pid % num_s_blocks

    offs_c = tl.arange(0, BLOCK_C)
    offs_s = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_c = offs_c < C
    mask_s = offs_s < S

    # x[n, c, s] => offset = n*C*S + c*S + s
    # 2D tile: [BLOCK_C, BLOCK_S]
    base = n * C * S
    ptrs = x_ptr + base + offs_c[:, None] * S + offs_s[None, :]
    mask = mask_c[:, None] & mask_s[None, :]

    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    # softmax along C (axis=0)
    x_max = tl.max(x, axis=0)  # [BLOCK_S]
    x_shift = x - x_max[None, :]
    e = tl.exp(x_shift)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)  # [BLOCK_S]
    sm = e / denom[None, :]

    # subtract per-channel param
    sub = tl.load(sub_ptr + offs_c, mask=mask_c, other=0.0)  # [BLOCK_C]
    y = sm - sub[:, None]
    # swish
    sig = 1.0 / (1.0 + tl.exp(-y))
    sw = y * sig
    sw = tl.where(mask, sw, -float('inf'))
    # max over C
    out_val = tl.max(sw, axis=0)  # [BLOCK_S]
    tl.store(out_ptr + n * S + offs_s, out_val, mask=mask_s)


def fused_post_conv(x, sub):
    # x: [N, C, D, H, W]
    N, C, D, H, W = x.shape
    S = D * H * W
    x_c = x.contiguous()
    out = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_S = 128
    num_s_blocks = (S + BLOCK_S - 1) // BLOCK_S
    grid = (N * num_s_blocks,)
    fused_softmax_sub_swish_max_kernel[grid](
        x_c, sub.contiguous(), out,
        N, C, S,
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding,
                 pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = fused_post_conv(x, self.subtract)
        return x