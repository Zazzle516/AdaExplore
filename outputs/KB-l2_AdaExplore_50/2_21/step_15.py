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
        triton.Config({'BLOCK': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
    ],
    key=['HW', 'CPG'],
)
@triton.jit
def fused_stats_kernel(
    x_ptr,           # [N, C, H, W]
    bias_ptr,        # [C]
    scale_ptr,       # [C]
    y_ptr,           # [N, C, H, W] sigmoid output
    mean_ptr,        # [N, G]
    rstd_ptr,        # [N, G]
    N, C, HW,
    G,
    CPG,
    eps,
    BLOCK: tl.constexpr,
    CPG_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_size = CPG * HW
    base = n * C * HW + g * CPG * HW

    sum_acc = tl.zeros([], dtype=tl.float32)
    sumsq_acc = tl.zeros([], dtype=tl.float32)

    num_chunks = (HW + BLOCK - 1) // BLOCK

    for ci_static in tl.static_range(0, CPG_C):
        c_valid = ci_static < CPG
        b_s = tl.load(bias_ptr + g * CPG + ci_static, mask=c_valid, other=0.0)
        s_s = tl.load(scale_ptr + g * CPG + ci_static, mask=c_valid, other=0.0)
        for k in range(0, num_chunks):
            offs = k * BLOCK + tl.arange(0, BLOCK)
            mask = (offs < HW) & c_valid
            x = tl.load(x_ptr + base + ci_static * HW + offs, mask=mask, other=0.0)
            y = tl.sigmoid((x + b_s) * s_s)
            tl.store(y_ptr + base + ci_static * HW + offs, y, mask=mask)
            y_masked = tl.where(mask, y, 0.0)
            sum_acc += tl.sum(y_masked)
            sumsq_acc += tl.sum(y_masked * y_masked)

    mean = sum_acc / group_size
    var = sumsq_acc / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + n * G + g, mean)
    tl.store(rstd_ptr + n * G + g, rstd)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=4, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def affine_apply_kernel(
    y_ptr,           # [N, C, H, W] sigmoid output
    gn_w_ptr,        # [C]
    gn_b_ptr,        # [C]
    mean_ptr,        # [N, G]
    rstd_ptr,        # [N, G]
    out_ptr,         # [N, C, H, W]
    N, C, HW, G, CPG,
    BLOCK: tl.constexpr,
):
    # one program per (n, c, chunk)
    n = tl.program_id(0)
    c = tl.program_id(1)
    k = tl.program_id(2)

    g = c // CPG

    w = tl.load(gn_w_ptr + c)
    b = tl.load(gn_b_ptr + c)
    mean = tl.load(mean_ptr + n * G + g)
    rstd = tl.load(rstd_ptr + n * G + g)

    a = w * rstd
    bb = b - mean * a

    base = n * C * HW + c * HW
    offs = k * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW
    y = tl.load(y_ptr + base + offs, mask=mask, other=0.0)
    z = y * a + bb
    tl.store(out_ptr + base + offs, z, mask=mask)


def fused_post_conv(x, bias, scale, gn_w, gn_b, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    CPG = C // num_groups
    HW = H * W
    CPG_C = triton.next_power_of_2(CPG)

    y = torch.empty_like(x)
    out = torch.empty_like(x)
    mean = torch.empty((N, num_groups), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N, num_groups), device=x.device, dtype=torch.float32)

    grid1 = (N * num_groups,)
    fused_stats_kernel[grid1](
        x, bias, scale, y, mean, rstd,
        N, C, HW, num_groups, CPG, eps,
        CPG_C=CPG_C,
    )

    grid2 = lambda meta: (N, C, (HW + meta['BLOCK'] - 1) // meta['BLOCK'])
    affine_apply_kernel[grid2](
        y, gn_w, gn_b, mean, rstd, out,
        N, C, HW, num_groups, CPG,
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