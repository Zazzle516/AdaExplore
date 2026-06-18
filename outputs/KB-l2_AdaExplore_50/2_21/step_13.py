import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
    ],
    key=['HW', 'CPG'],
)
@triton.jit
def fused_bias_scale_sigmoid_gn_kernel(
    x_ptr,           # [N, C, H, W]
    bias_ptr,        # [C]
    scale_ptr,       # [C]
    gn_w_ptr,        # [C]
    gn_b_ptr,        # [C]
    out_ptr,         # [N, C, H, W]
    N, C, HW,
    G,
    CPG,             # channels per group
    eps,
    BLOCK: tl.constexpr,
    CPG_C: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_size = CPG * HW

    # base offset for (n, g*CPG, 0, 0)
    base = n * C * HW + g * CPG * HW

    # load per-channel params for this group as vectors of length CPG_C
    c_offs = g * CPG + tl.arange(0, CPG_C)
    c_mask = tl.arange(0, CPG_C) < CPG
    bias_v = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0)
    scale_v = tl.load(scale_ptr + c_offs, mask=c_mask, other=0.0)
    gn_w_v = tl.load(gn_w_ptr + c_offs, mask=c_mask, other=0.0)
    gn_b_v = tl.load(gn_b_ptr + c_offs, mask=c_mask, other=0.0)

    num_chunks = (HW + BLOCK - 1) // BLOCK

    sum_acc = tl.zeros([], dtype=tl.float32)
    sumsq_acc = tl.zeros([], dtype=tl.float32)

    # First pass: accumulate stats only (no store), 1D loads per channel
    for ci_static in tl.static_range(0, CPG_C):
        c_valid = ci_static < CPG
        b_s = tl.load(bias_ptr + g * CPG + ci_static, mask=c_valid, other=0.0)
        s_s = tl.load(scale_ptr + g * CPG + ci_static, mask=c_valid, other=0.0)
        for k in range(0, num_chunks):
            offs = k * BLOCK + tl.arange(0, BLOCK)
            mask = (offs < HW) & c_valid
            x = tl.load(x_ptr + base + ci_static * HW + offs, mask=mask, other=0.0)
            y = tl.sigmoid((x + b_s) * s_s)
            y_masked = tl.where(mask, y, 0.0)
            sum_acc += tl.sum(y_masked)
            sumsq_acc += tl.sum(y_masked * y_masked)

    mean = sum_acc / group_size
    var = sumsq_acc / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: recompute sigmoid, normalize, write to out
    for ci_static in tl.static_range(0, CPG_C):
        c_valid = ci_static < CPG
        b_s = tl.load(bias_ptr + g * CPG + ci_static, mask=c_valid, other=0.0)
        s_s = tl.load(scale_ptr + g * CPG + ci_static, mask=c_valid, other=0.0)
        w_s = tl.load(gn_w_ptr + g * CPG + ci_static, mask=c_valid, other=0.0)
        bn_s = tl.load(gn_b_ptr + g * CPG + ci_static, mask=c_valid, other=0.0)
        for k in range(0, num_chunks):
            offs = k * BLOCK + tl.arange(0, BLOCK)
            mask = (offs < HW) & c_valid
            x = tl.load(x_ptr + base + ci_static * HW + offs, mask=mask, other=0.0)
            y = tl.sigmoid((x + b_s) * s_s)
            z = (y - mean) * rstd
            z = z * w_s + bn_s
            tl.store(out_ptr + base + ci_static * HW + offs, z, mask=mask)


def fused_post_conv(x, bias, scale, gn_w, gn_b, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    assert C % num_groups == 0
    CPG = C // num_groups
    out = torch.empty_like(x)

    HW = H * W
    CPG_C = triton.next_power_of_2(CPG)

    grid = (N * num_groups,)
    fused_bias_scale_sigmoid_gn_kernel[grid](
        x, bias, scale, gn_w, gn_b, out,
        N, C, HW,
        num_groups, CPG, eps,
        CPG_C=CPG_C,
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