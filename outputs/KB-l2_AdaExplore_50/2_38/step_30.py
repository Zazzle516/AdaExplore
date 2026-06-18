import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _clamp_softmax_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    S,  # spatial size = D*H*W
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # one program per (batch*channel) row
    row = tl.program_id(0)
    c = tl.program_id(1)  # channel id for scale lookup
    # actually program_id(0) is the flat (b,c) row
    pid = tl.program_id(0)
    # We launch grid = (B*C,) and pass C to recover channel
    # but simpler: grid = (B, C)
    b = tl.program_id(0)
    c = tl.program_id(1)
    C = tl.num_programs(1)
    row_start = (b * C + c) * S

    scale_val = tl.load(scale_ptr + c)

    # pass 1: compute max
    max_val = -float('inf')
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
        v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
        v = tl.where(mask, v, -float('inf'))
        m = tl.max(v, axis=0)
        max_val = tl.maximum(max_val, m)

    # pass 2: sum exp
    sum_val = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
        e = tl.exp(v - max_val)
        e = tl.where(mask, e, 0.0)
        sum_val += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_val

    # pass 3: write output
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
        e = tl.exp(v - max_val) * inv_sum * scale_val
        tl.store(out_ptr + row_start + idx, e, mask=mask)


def fused_clamp_softmax_scale(x, scale, clamp_min, clamp_max):
    B, C, D, H, W = x.shape
    S = D * H * W
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    scale_flat = scale.contiguous().view(-1)

    BLOCK = 1024
    if S <= 256:
        BLOCK = 256
    elif S <= 512:
        BLOCK = 512
    elif S <= 1024:
        BLOCK = 1024
    else:
        BLOCK = 1024

    grid = (B, C)
    _clamp_softmax_scale_kernel[grid](
        x_c, scale_flat, out,
        S,
        float(clamp_min), float(clamp_max),
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = fused_clamp_softmax_scale(x, self.scale, self.clamp_min, self.clamp_max)
        return x