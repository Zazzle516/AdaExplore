import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 32768}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 32768}, num_warps=16, num_stages=2),
    ],
    key=['HW', 'CPG_C'],
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
    CPG,             # channels per group (== CPG_C here)
    BLOCK: tl.constexpr,
    CPG_C: tl.constexpr,
    EPS: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_size = CPG_C * HW

    # base offset for (n, g*CPG, 0, 0)
    base = n * C * HW + g * CPG_C * HW

    # load bias and scale for this group's channels (CPG == CPG_C, no mask needed)
    c_arange = tl.arange(0, CPG_C)
    c_offs = g * CPG_C + c_arange
    bias_v = tl.load(bias_ptr + c_offs)
    scale_v = tl.load(scale_ptr + c_offs)
    gn_w_v = tl.load(gn_w_ptr + c_offs)
    gn_b_v = tl.load(gn_b_ptr + c_offs)

    # Hoist: precompute scale and bias*scale (FMA-style)
    scale_b = scale_v[:, None]
    bs_b = (bias_v * scale_v)[:, None]   # bias*scale
    # Hoist channel offset multiplication
    c_off2d = c_arange[:, None] * HW

    num_chunks = (HW + BLOCK - 1) // BLOCK

    # Pass 1: compute sigmoid(x*scale + bias*scale), accumulate stats. No scratch store.
    sum_acc = 0.0
    sumsq_acc = 0.0

    for k in range(0, num_chunks):
        offs = k * BLOCK + tl.arange(0, BLOCK)
        hw_mask = offs < HW
        ptrs = x_ptr + base + c_off2d + offs[None, :]
        x = tl.load(ptrs, mask=hw_mask[None, :], other=0.0)
        y = tl.sigmoid(x * scale_b + bs_b)
        y = tl.where(hw_mask[None, :], y, 0.0)
        sum_acc += tl.sum(y)
        sumsq_acc += tl.sum(y * y)

    mean = sum_acc / group_size
    var = sumsq_acc / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)

    w_b = gn_w_v[:, None]
    bn_b = gn_b_v[:, None]
    # fold: z = (y - mean) * rstd * w + bn = y * (rstd*w) + (bn - mean*rstd*w)
    scale_out = rstd * w_b
    bias_out = bn_b - mean * scale_out

    # Pass 2: recompute sigmoid from x_ptr (no scratch roundtrip), then normalize+affine.
    for k in range(0, num_chunks):
        offs = k * BLOCK + tl.arange(0, BLOCK)
        hw_mask = offs < HW
        ptrs = x_ptr + base + c_off2d + offs[None, :]
        x = tl.load(ptrs, mask=hw_mask[None, :], other=0.0)
        y = tl.sigmoid(x * scale_b + bs_b)
        z = y * scale_out + bias_out
        tl.store(out_ptr + base + c_off2d + offs[None, :], z, mask=hw_mask[None, :])


def fused_post_conv(x, bias, scale, gn_w, gn_b, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    assert C % num_groups == 0
    CPG = C // num_groups
    out = torch.empty_like(x)

    HW = H * W
    CPG_C = triton.next_power_of_2(CPG)
    assert CPG == CPG_C, "this fast path requires CPG to be a power of two"

    grid = (N * num_groups,)
    fused_bias_scale_sigmoid_gn_kernel[grid](
        x, bias, scale, gn_w, gn_b, out,
        N, C, HW,
        num_groups, CPG,
        CPG_C=CPG_C,
        EPS=float(eps),
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