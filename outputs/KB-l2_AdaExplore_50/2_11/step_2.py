import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
    ],
    key=['CPG', 'HW_OUT'],
)
@triton.jit
def fused_bn_tanh_pool_gn_kernel(
    x_ptr,           # input after conv_transpose: [N, C, H, W]
    out_ptr,         # output: [N, C, H/2, W/2]
    scale_ptr,       # bn fused scale [C]
    shift_ptr,       # bn fused shift [C]
    gn_weight_ptr,   # [C]
    gn_bias_ptr,     # [C]
    N, C, H, W,
    H_out, W_out,
    G,               # num_groups
    eps,
    CPG: tl.constexpr,
    HW_OUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # one program per (n, g)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    total = CPG * HW_OUT
    c_base = g * CPG

    # 2D tile: [CPG, BLOCK]
    offs_c = tl.arange(0, CPG)               # [CPG]
    offs_s = tl.arange(0, BLOCK)             # [BLOCK]
    mask_s = offs_s < HW_OUT                 # [BLOCK]

    # Load scale/shift per channel
    scale = tl.load(scale_ptr + c_base + offs_c)  # [CPG]
    shift = tl.load(shift_ptr + c_base + offs_c)  # [CPG]
    gw = tl.load(gn_weight_ptr + c_base + offs_c) # [CPG]
    gb = tl.load(gn_bias_ptr + c_base + offs_c)   # [CPG]

    oh = offs_s // W_out
    ow = offs_s % W_out
    ih = oh * 2
    iw = ow * 2

    # Compute input addresses for the 2D tile [CPG, BLOCK]
    # base offset per (c, s) in NCHW: n*C*H*W + c*H*W + ih*W + iw
    n_off = n * C * H * W
    c_stride = H * W
    c_off = (c_base + offs_c)[:, None] * c_stride  # [CPG, 1]

    s00 = (ih * W + iw)[None, :]              # [1, BLOCK]
    s01 = (ih * W + (iw + 1))[None, :]
    s10 = ((ih + 1) * W + iw)[None, :]
    s11 = ((ih + 1) * W + (iw + 1))[None, :]

    mask2 = mask_s[None, :]

    p00 = tl.load(x_ptr + n_off + c_off + s00, mask=mask2, other=-1e30)
    p01 = tl.load(x_ptr + n_off + c_off + s01, mask=mask2, other=-1e30)
    p10 = tl.load(x_ptr + n_off + c_off + s10, mask=mask2, other=-1e30)
    p11 = tl.load(x_ptr + n_off + c_off + s11, mask=mask2, other=-1e30)

    sc = scale[:, None]
    sh = shift[:, None]

    v00 = p00 * sc + sh
    v01 = p01 * sc + sh
    v10 = p10 * sc + sh
    v11 = p11 * sc + sh

    # tanh = 2*sigmoid(2x) - 1
    t00 = 2.0 * tl.sigmoid(2.0 * v00) - 1.0
    t01 = 2.0 * tl.sigmoid(2.0 * v01) - 1.0
    t10 = 2.0 * tl.sigmoid(2.0 * v10) - 1.0
    t11 = 2.0 * tl.sigmoid(2.0 * v11) - 1.0

    m0 = tl.maximum(t00, t01)
    m1 = tl.maximum(t10, t11)
    pooled = tl.maximum(m0, m1)  # [CPG, BLOCK]

    pooled_m = tl.where(mask2, pooled, 0.0)
    sum_val = tl.sum(pooled_m)
    sum_sq = tl.sum(pooled_m * pooled_m)

    mean = sum_val / total
    var = sum_sq / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    normed = (pooled - mean) * rstd * gw[:, None] + gb[:, None]

    # Store output [CPG, BLOCK]
    out_n_off = n * C * H_out * W_out
    out_c_stride = H_out * W_out
    out_c_off = (c_base + offs_c)[:, None] * out_c_stride
    out_s_off = offs_s[None, :]

    tl.store(out_ptr + out_n_off + out_c_off + out_s_off, normed, mask=mask2)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels

    def forward(self, x):
        # Conv transpose using torch (highly optimized)
        x = self.conv_transpose(x)

        # Fold BN
        bn = self.batch_norm
        if bn.training:
            # fall back
            x = bn(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        shift = bn.bias - bn.running_mean * scale

        N, C, H, W = x.shape
        H_out = H // 2
        W_out = W // 2
        G = self.num_groups
        CPG = C // G
        HW_OUT = H_out * W_out

        x = x.contiguous()
        out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        # BLOCK is next power of 2 >= HW_OUT
        BLOCK = 1
        while BLOCK < HW_OUT:
            BLOCK *= 2

        grid = (N * G,)
        fused_bn_tanh_pool_gn_kernel[grid](
            x, out,
            scale.contiguous(), shift.contiguous(),
            self.group_norm.weight.contiguous(), self.group_norm.bias.contiguous(),
            N, C, H, W,
            H_out, W_out,
            G,
            self.group_norm.eps,
            CPG=CPG,
            HW_OUT=HW_OUT,
            BLOCK=BLOCK,
        )
        return out