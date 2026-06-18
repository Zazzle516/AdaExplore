import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    S,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * S
    # scale indexed by channel = pid % C, but we'll pass it externally per row
    scale = tl.load(scale_ptr + pid)

    # Online softmax
    max_val = -float('inf')
    sum_exp = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        cur_max = tl.max(v, axis=0)
        new_max = tl.maximum(max_val, cur_max)
        e = tl.exp(v - new_max)
        e = tl.where(mask, e, 0.0)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(e, axis=0)
        max_val = new_max

    inv = 1.0 / sum_exp

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
        self.out_channels = out_channels
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        b, c, d, h, w = x.shape
        S = d * h * w
        x_flat = x.contiguous().view(b * c, S)
        out = torch.empty_like(x_flat)

        # Per-row scale: tile the scale (length C) across B rows
        scale_flat = self.scale.view(c).contiguous().repeat(b).contiguous()

        if S >= 4096:
            BLOCK = 4096
            num_warps = 8
        elif S >= 2048:
            BLOCK = 2048
            num_warps = 8
        elif S >= 1024:
            BLOCK = 1024
            num_warps = 8
        else:
            BLOCK = 512
            num_warps = 4

        grid = (b * c,)
        softmax_scale_kernel[grid](
            x_flat, scale_flat, out,
            S,
            self.clamp_min, self.clamp_max,
            BLOCK=BLOCK, num_warps=num_warps, num_stages=3,
        )
        return out.view(b, c, d, h, w)