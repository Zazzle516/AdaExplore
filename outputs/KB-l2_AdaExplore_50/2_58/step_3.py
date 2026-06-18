import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_lse_hardswish_bias_clamp_kernel(
    x_ptr,           # input after conv_transpose: (N, C, D, H, W)
    out_ptr,         # output: (N, 1, D, H, W)
    bias_val,        # scalar bias
    N, C, D, H, W,
    total_spatial,   # N*D*H*W
    BLOCK_SIZE: tl.constexpr,
    C_CONST: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_spatial

    # decode offs into (n, d, h, w)
    DHW = D * H * W
    HW = H * W
    n = offs // DHW
    rem = offs % DHW
    d = rem // HW
    rem2 = rem % HW
    h = rem2 // W
    w = rem2 % W

    # base offset into x for channel 0: n*C*DHW + d*HW + h*W + w
    base = n * (C_CONST * DHW) + d * HW + h * W + w
    channel_stride = DHW

    # first pass: max over channels
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)
    for c in tl.static_range(0, C_CONST):
        ptr = base + c * channel_stride
        v = tl.load(x_ptr + ptr, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, v)

    # second pass: sum exp
    sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for c in tl.static_range(0, C_CONST):
        ptr = base + c * channel_stride
        v = tl.load(x_ptr + ptr, mask=mask, other=0.0)
        sum_exp += tl.exp(v - max_val)

    lse = max_val + tl.log(sum_exp)

    # hardswish: x * sigmoid(x+3) / 6
    hs = lse * tl.sigmoid(lse + 3.0) / 6.0

    # subtract bias
    res = hs - bias_val

    # clamp [-1, 1]
    res = tl.minimum(tl.maximum(res, -1.0), 1.0)

    tl.store(out_ptr + offs, res, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape
        x = x.contiguous()
        out = torch.empty((N, 1, D, H, W), device=x.device, dtype=x.dtype)
        total_spatial = N * D * H * W
        bias_val = float(self.bias.item())

        BLOCK_SIZE = 256
        grid = ((total_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        fused_lse_hardswish_bias_clamp_kernel[grid](
            x, out, bias_val,
            N, C, D, H, W,
            total_spatial,
            BLOCK_SIZE=BLOCK_SIZE,
            C_CONST=C,
            num_warps=4,
        )
        return out