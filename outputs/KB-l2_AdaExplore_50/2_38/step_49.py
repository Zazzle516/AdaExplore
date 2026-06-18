import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK': 32768}, num_warps=32, num_stages=2),
    ],
    key=['S'],
)
@triton.jit
def softmax_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    B, C, S,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    row_start = (b * C + c) * S
    scale = tl.load(scale_ptr + c)

    # Since output of clamp(x, 0, 1) is in [0, 1], max is at most clamp_max.
    # We use the fact that max <= clamp_max for online softmax stability.
    max_val: tl.constexpr = clamp_max

    sum_val = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        x = tl.minimum(tl.maximum(x, clamp_min), clamp_max)
        e = tl.exp(x - max_val)
        e = tl.where(mask, e, 0.0)
        sum_val += tl.sum(e, axis=0)

    coef = (1.0 / sum_val) * scale

    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        x = tl.minimum(tl.maximum(x, clamp_min), clamp_max)
        y = tl.exp(x - max_val) * coef
        tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        try:
            self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last_3d)
        except Exception:
            pass
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = x.contiguous()
        b, c, d, h, w = x.shape
        S = d * h * w

        x_flat = x.view(b, c, S)
        out = torch.empty_like(x_flat)

        scale_flat = self.scale.view(c).contiguous()

        grid = (b * c,)
        softmax_scale_kernel[grid](
            x_flat, scale_flat, out,
            b, c, S,
            self.clamp_min, self.clamp_max,
        )

        return out.view(b, c, d, h, w)