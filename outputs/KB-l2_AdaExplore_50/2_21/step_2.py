import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------
# Fused Conv2d (3x3, stride=1, no padding) + bias + scale + sigmoid
# Implemented as im2col-style GEMM. One program per (n, oc-tile, spatial-tile).
# Output written in NCHW.
# ---------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'IC', 'OH', 'OW'],
)
@triton.jit
def conv3x3_bias_scale_sigmoid_kernel(
    x_ptr, w_ptr, conv_b_ptr, bias_ptr, scale_ptr, out_ptr,
    N, IC, H, W, OC, OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
    K: tl.constexpr,        # IC*9
):
    pid = tl.program_id(0)
    n   = tl.program_id(1)

    num_m = tl.cdiv(OC, BLOCK_M)
    num_n = tl.cdiv(OH * OW, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial index in OH*OW

    mask_m = offs_m < OC
    mask_n = offs_n < OH * OW

    oh = offs_n // OW
    ow = offs_n % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = IC * 9; iterate
    for k in range(0, K):
        ic = k // 9
        kk = k - ic * 9
        kh = kk // 3
        kw = kk - kh * 3

        ih = oh + kh
        iw = ow + kw

        x_off = n * stride_xn + ic * stride_xc + ih * stride_xh + iw * stride_xw
        x_vals = tl.load(x_ptr + x_off, mask=mask_n, other=0.0)  # [BLOCK_N]

        w_off = offs_m * K + k
        w_vals = tl.load(w_ptr + w_off, mask=mask_m, other=0.0)  # [BLOCK_M]

        acc += w_vals[:, None] * x_vals[None, :]

    # Add conv bias
    cb = tl.load(conv_b_ptr + offs_m, mask=mask_m, other=0.0)
    acc += cb[:, None]
    # Add extra bias
    eb = tl.load(bias_ptr + offs_m, mask=mask_m, other=0.0)
    acc += eb[:, None]
    # Multiply scale
    sc = tl.load(scale_ptr + offs_m, mask=mask_m, other=0.0)
    acc *= sc[:, None]
    # Sigmoid
    acc = tl.sigmoid(acc)

    # Store NCHW: out[n, oc, oh, ow]
    out_off = n * stride_on + offs_m[:, None] * stride_oc + oh[None, :] * stride_oh + ow[None, :] * stride_ow
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


# ---------------------------------------------------------------------
# GroupNorm kernel: one program per (n, group); Welford-free 2-pass.
# ---------------------------------------------------------------------
@triton.jit
def groupnorm_kernel(
    x_ptr, out_ptr, gn_w_ptr, gn_b_ptr,
    N, C, SPATIAL,
    GROUPS: tl.constexpr, CH_PER_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    base = n * C * SPATIAL + g * CH_PER_GROUP * SPATIAL

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    total_elems = GROUP_SIZE
    for off in range(0, total_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < total_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)

    mean = sum_val / total_elems
    var = sum_sq / total_elems - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, total_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < total_elems
        # determine channel within group
        ci = idx // SPATIAL  # [0, CH_PER_GROUP)
        c = g * CH_PER_GROUP + ci
        gw = tl.load(gn_w_ptr + c, mask=mask, other=0.0)
        gb = tl.load(gn_b_ptr + c, mask=mask, other=0.0)
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        y = (x - mean) * rstd
        y = y * gw + gb
        tl.store(out_ptr + base + idx, y, mask=mask)


def fused_conv_bsg(x, w, conv_b, bias, scale):
    N, IC, H, W = x.shape
    OC, _, KH, KW = w.shape
    assert KH == 3 and KW == 3
    OH = H - 2
    OW = W - 2
    K = IC * 9

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    w_flat = w.reshape(OC, K).contiguous()

    def grid(meta):
        return (
            triton.cdiv(OC, meta['BLOCK_M']) * triton.cdiv(OH * OW, meta['BLOCK_N']),
            N,
        )

    conv3x3_bias_scale_sigmoid_kernel[grid](
        x, w_flat, conv_b, bias, scale, out,
        N, IC, H, W, OC, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        K=K,
    )
    return out


def do_groupnorm(x, gn_w, gn_b, num_groups, eps):
    N, C, H, W = x.shape
    SPATIAL = H * W
    CH_PER_GROUP = C // num_groups
    GROUP_SIZE = CH_PER_GROUP * SPATIAL

    out = torch.empty_like(x)
    BLOCK_SIZE = 1024

    grid = (N * num_groups,)
    groupnorm_kernel[grid](
        x, out, gn_w, gn_b,
        N, C, SPATIAL,
        GROUPS=num_groups, CH_PER_GROUP=CH_PER_GROUP,
        GROUP_SIZE=GROUP_SIZE,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
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
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous()
        conv_b = self.conv.bias.contiguous()
        bias_flat = self.bias.view(-1).contiguous()
        scale_flat = self.scale.view(-1).contiguous()

        y = fused_conv_bsg(x, w, conv_b, bias_flat, scale_flat)

        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps

        out = do_groupnorm(y, gn_w, gn_b, self.num_groups, eps)
        return out