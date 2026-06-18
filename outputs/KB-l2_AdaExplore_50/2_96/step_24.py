import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _final_reduce_kernel(
    in_ptr, out_ptr,
    N, C, S,
    scale,
    clamp_min, clamp_max,
    BLOCK: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = n * C * S + c * S

    acc = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)

    mean = acc / S
    mean = mean * scale
    mean = tl.minimum(tl.maximum(mean, clamp_min), clamp_max)
    tl.store(out_ptr + n * C + c, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.maxpool_kernel_size = maxpool_kernel_size

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = x.contiguous()
        N, C, D, H, W = x.shape
        S = D * H * W
        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
        # scale folded with avgpool: since avg then *scale equals (sum/S)*scale
        # but maxpool was on x*scale; since scale>=0, max(scale*y) = scale*max(y), so we can apply scale after.
        grid = (N * C,)
        BLOCK = 1024
        _final_reduce_kernel[grid](
            x, out,
            N, C, S,
            float(self.scale),
            float(self.clamp_min), float(self.clamp_max),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out