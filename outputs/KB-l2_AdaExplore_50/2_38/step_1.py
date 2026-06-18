import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    B, C, S,
    clamp_min, clamp_max,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    row_start = (b * C + c) * S
    scale = tl.load(scale_ptr + c)

    # First pass: find max
    max_val = -float('inf')
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        cur = tl.max(v, axis=0)
        max_val = tl.maximum(max_val, cur)

    # Second pass: sum exp
    sum_exp = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        e = tl.exp(v - max_val)
        e = tl.where(mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    inv = 1.0 / sum_exp

    # Third pass: write output
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        e = tl.exp(v - max_val) * inv * scale
        tl.store(out_ptr + row_start + idx, e, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        b, c, d, h, w = x.shape
        S = d * h * w
        x_flat = x.contiguous().view(b * c, S)
        out = torch.empty_like(x_flat)
        scale_flat = self.scale.view(-1).contiguous()

        # Choose BLOCK size
        if S >= 1024:
            BLOCK = 1024
            num_warps = 8
        elif S >= 512:
            BLOCK = 512
            num_warps = 4
        else:
            BLOCK = 256
            num_warps = 4

        grid = (b * c,)
        softmax_scale_kernel[grid](
            x_flat, scale_flat, out,
            b, c, S,
            self.clamp_min, self.clamp_max,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(b, c, d, h, w)