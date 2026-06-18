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
        triton.Config({'BLOCK': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
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
    EPS: tl.constexpr,
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
    c_offs = g * CPG + c_arange
    bias_v = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0)
    scale_v = tl.load(scale_ptr + c_offs, mask=c_mask, other=0.0)
    gn_w_v = tl.load(gn_w_ptr + c_offs, mask=c_mask, other=0.0)
    gn_b_v = tl.load(gn_b_ptr + c_offs, mask=c_mask, other=0.0)

    bias_b = bias_v[:, None]
    scale_b = scale_v[:, None]

    num_chunks = (HW + BLOCK - 1) // BLOCK

    # Pass 1: compute stats only (no intermediate store)
    sum_acc = 0.0
    sumsq_acc = 0.0

    for k in range(0, num_chunks):
        offs = k * BLOCK + tl.arange(0, BLOCK)
        hw_mask = offs < HW
        mask2d = c_mask[:, None] & hw_mask[None, :]
        ptrs = x_ptr + base + c_arange[:, None] * HW + offs[None, :]
        x = tl.load(ptrs, mask=mask2d, other=0.0)
        y = (x + bias_b) * scale_b
        y = tl.sigmoid(y)
        y = tl.where(mask2d, y, 0.0)
        sum_acc += tl.sum(y)
        sumsq_acc += tl.sum(y * y)

    mean = sum_acc / group_size
    var = sumsq_acc / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)

    w_b = gn_w_v[:, None]
    bn_b = gn_b_v[:, None]
    # Fold normalization into affine: z = (y - mean) * rstd * w + b = y * (rstd*w) + (b - mean*rstd*w)
    a = rstd * w_b
    c_term = bn_b - mean * a

    # Pass 2: recompute y from x, normalize, store
    for k in range(0, num_chunks):
        offs = k * BLOCK + tl.arange(0, BLOCK)
        hw_mask = offs < HW
        mask2d = c_mask[:, None] & hw_mask[None, :]
        ptrs = x_ptr + base + c_arange[:, None] * HW + offs[None, :]
        x = tl.load(ptrs, mask=mask2d, other=0.0)
        y = (x + bias_b) * scale_b
        y = tl.sigmoid(y)
        z = y * a + c_term
        tl.store(out_ptr + base + c_arange[:, None] * HW + offs[None, :], z, mask=mask2d)


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
        num_groups, CPG,
        EPS=float(eps),
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
        x = x.contiguous()
        bias_flat = self.bias.view(-1).contiguous()
        scale_flat = self.scale.view(-1).contiguous()
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps
        out = fused_post_conv(x, bias_flat, scale_flat, gn_w, gn_b, self.num_groups, eps)
        return out