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
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # x[n, c, s] -> base = n*C*S + c*S + s
    # Load (BLOCK_C, BLOCK_S)
    ptrs = x_ptr + pid_n * C * S + offs_c[:, None] * S + offs_s[None, :]
    mask = mask_c[:, None] & mask_s[None, :]
    x = tl.load(ptrs, mask=mask, other=-float('inf'))

    # softmax over C
    m = tl.max(x, axis=0)  # (BLOCK_S,)
    e = tl.exp(x - m[None, :])
    e = tl.where(mask_c[:, None], e, 0.0)
    s_sum = tl.sum(e, axis=0)  # (BLOCK_S,)
    sm = e / s_sum[None, :]

    sub = tl.load(sub_ptr + offs_c, mask=mask_c, other=0.0)
    y = sm - sub[:, None]
    sw = y * tl.sigmoid(y)
    sw = tl.where(mask_c[:, None], sw, -float('inf'))
    out_val = tl.max(sw, axis=0)  # (BLOCK_S,)

    out_ptrs = out_ptr + pid_n * S + offs_s
    tl.store(out_ptrs, out_val, mask=mask_s)


def fused_post_conv(x, sub):
    N, C, D, H, W = x.shape
    S = D * H * W
    x_c = x.contiguous()
    out = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_S = 128
    grid = (N, (S + BLOCK_S - 1) // BLOCK_S)
    fused_softmax_sub_swish_max_kernel[grid](
        x_c, sub.contiguous(), out,
        N, C, S,
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = fused_post_conv(x, self.subtract)
        return x