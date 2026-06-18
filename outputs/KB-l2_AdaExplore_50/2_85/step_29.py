import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_stats_kernel(
    conv_ptr, mean_ptr, rstd_ptr,
    N, OC, SPATIAL, GROUPS, CH_PER_GROUP,
    eps,
    BLOCK_SP: tl.constexpr,
    CH_PER_GROUP_C: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)

    group_size = CH_PER_GROUP_C * SPATIAL
    base = (n * OC + g * CH_PER_GROUP_C) * SPATIAL

    sum_v = 0.0
    sumsq_v = 0.0
    for c in tl.static_range(CH_PER_GROUP_C):
        for sp_start in range(0, SPATIAL, BLOCK_SP):
            offs = sp_start + tl.arange(0, BLOCK_SP)
            m = offs < SPATIAL
            v = tl.load(conv_ptr + base + c * SPATIAL + offs, mask=m, other=0.0)
            sum_v += tl.sum(tl.where(m, v, 0.0), axis=0)
            sumsq_v += tl.sum(tl.where(m, v * v, 0.0), axis=0)

    mean = sum_v / group_size
    var = sumsq_v / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + n * GROUPS + g, mean)
    tl.store(rstd_ptr + n * GROUPS + g, rstd)


@triton.jit
def fused_epilogue_kernel(
    conv_ptr, mean_ptr, rstd_ptr, gn_w_ptr, gn_b_ptr, scale_ptr, out_ptr,
    N, OC, OH, OW, POH, POW, GROUPS, CH_PER_GROUP,
    clamp_min, clamp_max,
    pool_k: tl.constexpr,
    BLOCK_PH: tl.constexpr,
    BLOCK_PW: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    pt = tl.program_id(2)

    # tile over pool grid
    num_tiles_w = (POW + BLOCK_PW - 1) // BLOCK_PW
    tile_h = pt // num_tiles_w
    tile_w = pt % num_tiles_w

    ph_off = tile_h * BLOCK_PH + tl.arange(0, BLOCK_PH)  # [BLOCK_PH]
    pw_off = tile_w * BLOCK_PW + tl.arange(0, BLOCK_PW)  # [BLOCK_PW]
    ph_mask = ph_off < POH
    pw_mask = pw_off < POW
    tile_mask = ph_mask[:, None] & pw_mask[None, :]

    g = c // CH_PER_GROUP
    mean = tl.load(mean_ptr + n * GROUPS + g)
    rstd = tl.load(rstd_ptr + n * GROUPS + g)
    gn_w = tl.load(gn_w_ptr + c)
    gn_b = tl.load(gn_b_ptr + c)
    sc = tl.load(scale_ptr + c)

    # combined affine: out = (v - mean)*rstd*gn_w*sc + gn_b*sc
    a = rstd * gn_w * sc
    b = gn_b * sc - mean * a

    OHW = OH * OW
    conv_base = (n * OC + c) * OHW

    neg_inf = -float('inf')
    max_val = tl.full([BLOCK_PH, BLOCK_PW], neg_inf, dtype=tl.float32)

    for pkh in tl.static_range(pool_k):
        for pkw in tl.static_range(pool_k):
            oh = ph_off[:, None] * pool_k + pkh  # [BLOCK_PH, 1]
            ow = pw_off[None, :] * pool_k + pkw  # [1, BLOCK_PW]
            valid = tile_mask & (oh < OH) & (ow < OW)
            idx = oh * OW + ow
            v = tl.load(conv_ptr + conv_base + idx, mask=valid, other=0.0)
            scaled = v * a + b
            scaled = tl.where(valid, scaled, neg_inf)
            max_val = tl.maximum(max_val, scaled)

    out_v = tl.minimum(tl.maximum(max_val, clamp_min), clamp_max)

    out_base = (n * OC + c) * (POH * POW)
    out_idx = ph_off[:, None] * POW + pw_off[None, :]
    tl.store(out_ptr + out_base + out_idx, out_v, mask=tile_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.maxpool_kernel_size = maxpool_kernel_size
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        pool_k = self.maxpool_kernel_size
        POH = OH // pool_k
        POW = OW // pool_k

        CH_PER_GROUP = OC // self.num_groups

        conv_out = F.conv2d(x, self.conv.weight, self.conv.bias)

        scale_flat = self.scale.reshape(-1).contiguous()
        if scale_flat.numel() != OC:
            scale_flat = self.scale.expand(OC, 1, 1).reshape(-1).contiguous()

        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()

        SPATIAL = OH * OW
        mean = torch.empty((N, self.num_groups), device=x.device, dtype=torch.float32)
        rstd = torch.empty((N, self.num_groups), device=x.device, dtype=torch.float32)

        BLOCK_SP = 1024
        gn_stats_kernel[(N, self.num_groups)](
            conv_out, mean, rstd,
            N, OC, SPATIAL, self.num_groups, CH_PER_GROUP,
            self.eps,
            BLOCK_SP=BLOCK_SP,
            CH_PER_GROUP_C=CH_PER_GROUP,
            num_warps=8,
            num_stages=3,
        )

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

        BLOCK_PH = 8
        BLOCK_PW = 16
        num_tiles_h = (POH + BLOCK_PH - 1) // BLOCK_PH
        num_tiles_w = (POW + BLOCK_PW - 1) // BLOCK_PW
        grid = (N, OC, num_tiles_h * num_tiles_w)

        fused_epilogue_kernel[grid](
            conv_out, mean, rstd, gn_w, gn_b, scale_flat, out,
            N, OC, OH, OW, POH, POW, self.num_groups, CH_PER_GROUP,
            self.clamp_min, self.clamp_max,
            pool_k=pool_k,
            BLOCK_PH=BLOCK_PH,
            BLOCK_PW=BLOCK_PW,
            num_warps=4,
            num_stages=2,
        )

        return out