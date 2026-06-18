import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=3),
    ],
    key=['S', 'C'],
)
@triton.jit
def _fused_kernel(
    x_ptr, scale_ptr, out_ptr,
    S, C,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bc = tl.program_id(0)
    c = bc % C
    row_start = bc * S

    NEG_INF = float('-inf')

    # Pass 1 (online): track running max and running sum of exp(v - max)
    running_max = NEG_INF
    running_sum = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=NEG_INF)
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        v = tl.where(mask, v, NEG_INF)
        block_max = tl.max(v, axis=0)
        new_max = tl.maximum(running_max, block_max)
        # rescale running_sum
        running_sum = running_sum * tl.exp(running_max - new_max)
        e = tl.exp(v - new_max)
        e = tl.where(mask, e, 0.0)
        running_sum += tl.sum(e, axis=0)
        running_max = new_max

    inv_sum = 1.0 / running_sum
    s = tl.load(scale_ptr + c)

    # Pass 2: write softmax * scale
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        e = tl.exp(v - running_max) * inv_sum * s
        tl.store(out_ptr + row_start + idx, e, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.out_channels = out_channels

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        b, c, d, h, w = x.shape
        S = d * h * w
        x_flat = x.contiguous().view(b * c, S)
        out = torch.empty_like(x_flat)
        scale_flat = self.scale.view(-1).contiguous()

        grid = (b * c,)
        _fused_kernel[grid](
            x_flat, scale_flat, out,
            S, c,
            self.clamp_min, self.clamp_max,
        )
        return out.view(b, c, d, h, w)