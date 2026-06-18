import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_fused_epilogue_kernel(
    x_ptr, w_ptr, cb_ptr, bias_ptr, scale_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # program ids: (n, oc_tile, sp_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    oh = sp_offs // OW
    ow = sp_offs % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # K dimension = IC * KH * KW
    # We iterate over KH, KW, IC as constexpr loops for unroll
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # no padding
            iw = ow + kw
            # input pointers: x[n, ic, ih, iw]
            # spatial linear offset within (ic) plane
            x_sp_off = ih * IW + iw  # [BLOCK_SP]
            for ic in tl.static_range(0, IC_C):
                # load x[n, ic, ih, iw] for sp tile -> shape [BLOCK_SP]
                x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + x_sp_off
                x_val = tl.load(x_ptr + x_off, mask=sp_mask, other=0.0)  # [BLOCK_SP]
                # load w[oc, ic, kh, kw] -> shape [BLOCK_OC]
                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                acc += w_val[:, None] * x_val[None, :]

    # Epilogue: + conv_bias + extra_bias, * scale, sigmoid
    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)  # conv bias
    eb = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)  # extra bias
    sc = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    bias_total = cb + eb  # [BLOCK_OC]

    acc = (acc + bias_total[:, None]) * sc[:, None]
    acc = tl.sigmoid(acc)

    # Store: out[n, oc, sp]
    out_off = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
    ],
    key=['SPATIAL', 'CH_PER_GROUP'],
)
@triton.jit
def groupnorm_kernel(
    x_ptr, out_ptr,
    gn_weight_ptr, gn_bias_ptr,
    C, GROUPS, CH_PER_GROUP: tl.constexpr,
    eps,
    SPATIAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    base = n * C * SPATIAL + g * CH_PER_GROUP * SPATIAL
    total = GROUP_SIZE

    sum_val = 0.0
    sum_sq = 0.0

    for ci in tl.static_range(0, CH_PER_GROUP):
        ch_base = base + ci * SPATIAL
        for off in range(0, SPATIAL, BLOCK_SIZE):
            idx = off + tl.arange(0, BLOCK_SIZE)
            mask = idx < SPATIAL
            x = tl.load(x_ptr + ch_base + idx, mask=mask, other=0.0)
            x = tl.where(mask, x, 0.0)
            sum_val += tl.sum(x)
            sum_sq += tl.sum(x * x)

    mean = sum_val / total
    var = sum_sq / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for ci in tl.static_range(0, CH_PER_GROUP):
        c = g * CH_PER_GROUP + ci
        gw = tl.load(gn_weight_ptr + c)
        gb = tl.load(gn_bias_ptr + c)
        ch_base = base + ci * SPATIAL
        for off in range(0, SPATIAL, BLOCK_SIZE):
            idx = off + tl.arange(0, BLOCK_SIZE)
            mask = idx < SPATIAL
            x = tl.load(x_ptr + ch_base + idx, mask=mask, other=0.0)
            y = (x - mean) * rstd
            y = y * gw + gb
            tl.store(out_ptr + ch_base + idx, y, mask=mask)


def fused_conv_epilogue(x, weight, conv_bias, extra_bias, scale):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 128

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

    conv_fused_epilogue_kernel[grid](
        x, weight, conv_bias, extra_bias, scale, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH=KH, KW=KW,
        IC_C=IC,
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out


def groupnorm_only(x, gn_weight, gn_bias, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    SPATIAL = H * W
    CH_PER_GROUP = C // num_groups
    GROUP_SIZE = CH_PER_GROUP * SPATIAL

    out = torch.empty_like(x)
    grid = (N * num_groups,)
    groupnorm_kernel[grid](
        x, out,
        gn_weight, gn_bias,
        C, num_groups, CH_PER_GROUP,
        eps,
        SPATIAL=SPATIAL,
        GROUP_SIZE=GROUP_SIZE,
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
        self.in_channels = in_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()
        eb = self.bias.view(-1).contiguous()
        sc = self.scale.view(-1).contiguous()

        y = fused_conv_epilogue(x, w, cb, eb, sc)

        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps
        out = groupnorm_only(y, gn_w, gn_b, self.num_groups, eps)
        return out