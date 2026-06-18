import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=2),
    ],
    key=['HW', 'CPG'],
)
@triton.jit
def fused_bias_scale_sigmoid_gn_kernel(
    x_ptr,
    bias_ptr,
    scale_ptr,
    gn_w_ptr,
    gn_b_ptr,
    out_ptr,
    N, C, HW,
    G,
    CPG: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_size = CPG * HW
    base = n * C * HW + g * CPG * HW

    num_chunks = (HW + BLOCK - 1) // BLOCK

    sum_acc = tl.zeros([], dtype=tl.float32)
    sumsq_acc = tl.zeros([], dtype=tl.float32)

    # First pass: accumulate stats
    for ci_static in tl.static_range(0, CPG):
        b_s = tl.load(bias_ptr + g * CPG + ci_static)
        s_s = tl.load(scale_ptr + g * CPG + ci_static)
        for k in range(0, num_chunks):
            offs = k * BLOCK + tl.arange(0, BLOCK)
            mask = offs < HW
            x = tl.load(x_ptr + base + ci_static * HW + offs, mask=mask, other=0.0)
            y = tl.sigmoid((x + b_s) * s_s)
            y_masked = tl.where(mask, y, 0.0)
            sum_acc += tl.sum(y_masked)
            sumsq_acc += tl.sum(y_masked * y_masked)

    mean = sum_acc / group_size
    var = sumsq_acc / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: write output with fused affine
    for ci_static in tl.static_range(0, CPG):
        b_s = tl.load(bias_ptr + g * CPG + ci_static)
        s_s = tl.load(scale_ptr + g * CPG + ci_static)
        w_s = tl.load(gn_w_ptr + g * CPG + ci_static)
        bn_s = tl.load(gn_b_ptr + g * CPG + ci_static)
        a = rstd * w_s
        b_off = bn_s - mean * a
        for k in range(0, num_chunks):
            offs = k * BLOCK + tl.arange(0, BLOCK)
            mask = offs < HW
            x = tl.load(x_ptr + base + ci_static * HW + offs, mask=mask, other=0.0)
            y = tl.sigmoid((x + b_s) * s_s)
            z = y * a + b_off
            tl.store(out_ptr + base + ci_static * HW + offs, z, mask=mask)


def fused_post_conv(x, bias, scale, gn_w, gn_b, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    CPG = C // num_groups
    out = torch.empty_like(x)

    HW = H * W

    grid = (N * num_groups,)
    fused_bias_scale_sigmoid_gn_kernel[grid](
        x, bias, scale, gn_w, gn_b, out,
        N, C, HW,
        num_groups, CPG, eps,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        bias_flat = self.bias.view(-1).contiguous()
        scale_flat = self.scale.view(-1).contiguous()
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps
        x = x.contiguous()
        out = fused_post_conv(x, bias_flat, scale_flat, gn_w, gn_b, self.num_groups, eps)
        return out