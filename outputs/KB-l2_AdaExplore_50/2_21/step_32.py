import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=3),
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

    c_arange = tl.arange(0, CPG_C)
    c_mask = c_arange < CPG
    bias_v = tl.load(bias_ptr + g * CPG + c_arange, mask=c_mask, other=0.0)
    scale_v = tl.load(scale_ptr + g * CPG + c_arange, mask=c_mask, other=0.0)
    gn_w_v = tl.load(gn_w_ptr + g * CPG + c_arange, mask=c_mask, other=0.0)
    gn_b_v = tl.load(gn_b_ptr + g * CPG + c_arange, mask=c_mask, other=0.0)

    num_chunks = (HW + BLOCK - 1) // BLOCK

    sum_acc = tl.zeros([], dtype=tl.float32)
    sumsq_acc = tl.zeros([], dtype=tl.float32)

    # First pass: compute sigmoid, store to out, accumulate stats
    for k in range(0, num_chunks):
        offs = k * BLOCK + tl.arange(0, BLOCK)
        sp_mask = offs < HW
        # 2D tile: [CPG_C, BLOCK]
        ptrs = base + c_arange[:, None] * HW + offs[None, :]
        tile_mask = c_mask[:, None] & sp_mask[None, :]
        x = tl.load(x_ptr + ptrs, mask=tile_mask, other=0.0)
        y = tl.sigmoid((x + bias_v[:, None]) * scale_v[:, None])
        y = tl.where(tile_mask, y, 0.0)
        tl.store(out_ptr + ptrs, y, mask=tile_mask)
        sum_acc += tl.sum(y)
        sumsq_acc += tl.sum(y * y)

    mean = sum_acc / group_size
    var = sumsq_acc / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    a_v = rstd * gn_w_v
    b_off_v = gn_b_v - mean * a_v

    # Second pass: read sigmoid output, apply affine, write back
    for k in range(0, num_chunks):
        offs = k * BLOCK + tl.arange(0, BLOCK)
        sp_mask = offs < HW
        ptrs = base + c_arange[:, None] * HW + offs[None, :]
        tile_mask = c_mask[:, None] & sp_mask[None, :]
        y = tl.load(out_ptr + ptrs, mask=tile_mask, other=0.0)
        z = y * a_v[:, None] + b_off_v[:, None]
        tl.store(out_ptr + ptrs, z, mask=tile_mask)


def fused_post_conv(x, bias, scale, gn_w, gn_b, num_groups, eps=1e-5):
    N, C, H, W = x.shape
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