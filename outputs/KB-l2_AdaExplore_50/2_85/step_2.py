import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W, OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_PIX: tl.constexpr,
):
    # grid: (N, OC // BLOCK_OC, ceil(OH*OW / BLOCK_PIX))
    n = tl.program_id(0)
    oc_block = tl.program_id(1)
    pix_block = tl.program_id(2)

    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    pix_offs = pix_block * BLOCK_PIX + tl.arange(0, BLOCK_PIX)  # [BLOCK_PIX]
    pix_mask = pix_offs < (OH * OW)
    oc_mask = oc_offs < OC

    oh = pix_offs // OW
    ow = pix_offs % OW

    acc = tl.zeros([BLOCK_PIX, BLOCK_OC], dtype=tl.float32)

    for ic in tl.static_range(IC_C):
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                ih = oh + kh
                iw = ow + kw
                x_idx = ((n * IC + ic) * H + ih) * W + iw
                x_val = tl.load(x_ptr + x_idx, mask=pix_mask, other=0.0)
                w_idx = ((oc_offs * IC + ic) * KH + kh) * KW + kw
                w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)
                acc += x_val[:, None] * w_val[None, :]

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_val[None, :]

    # store as [N, OC, OH*OW]
    out_idx = (n * OC + oc_offs[None, :]) * (OH * OW) + pix_offs[:, None]
    mask = pix_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=mask)


@triton.jit
def gn_scale_pool_clamp_kernel(
    in_ptr, gn_w_ptr, gn_b_ptr, scale_ptr, out_ptr,
    N, OC, OH, OW, POH, POW,
    GROUPS: tl.constexpr,
    CH_PER_GROUP: tl.constexpr,
    SPATIAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # CH_PER_GROUP * SPATIAL
    POOL_K: tl.constexpr,
    POH_C: tl.constexpr,
    POW_C: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    eps: tl.constexpr,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
):
    # grid: (N, GROUPS)
    n = tl.program_id(0)
    g = tl.program_id(1)

    # Compute mean/var over group: CH_PER_GROUP channels x SPATIAL pixels
    # input layout [N, OC, OH*OW]
    base = (n * OC + g * CH_PER_GROUP) * SPATIAL  # ptr to start of group

    sum_acc = tl.zeros([1], dtype=tl.float32)
    sumsq_acc = tl.zeros([1], dtype=tl.float32)

    # iterate over channels, then spatial blocks
    for c in tl.static_range(CH_PER_GROUP):
        c_base = base + c * SPATIAL
        for sp_start in range(0, SPATIAL, BLOCK_SP):
            sp_offs = sp_start + tl.arange(0, BLOCK_SP)
            sp_mask = sp_offs < SPATIAL
            v = tl.load(in_ptr + c_base + sp_offs, mask=sp_mask, other=0.0)
            sum_acc += tl.sum(v, axis=0)
            sumsq_acc += tl.sum(v * v, axis=0)

    total_count = GROUP_SIZE
    mean = sum_acc / total_count
    var = sumsq_acc / total_count - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Now for each channel in group, do GN+scale+pool+clamp
    # Output layout [N, OC, POH*POW]
    NUM_POOL: tl.constexpr = POH_C * POW_C

    for c in tl.static_range(CH_PER_GROUP):
        oc = g * CH_PER_GROUP + c
        gn_w = tl.load(gn_w_ptr + oc)
        gn_b = tl.load(gn_b_ptr + oc)
        sc = tl.load(scale_ptr + oc)
        # combined affine: y = (x - mean) * rstd * gn_w * sc + gn_b * sc
        a = rstd * gn_w * sc
        bsh = gn_b * sc
        c_in_base = base + c * SPATIAL
        c_out_base = (n * OC + oc) * NUM_POOL

        # iterate pool positions
        pool_offs = tl.arange(0, 16)  # placeholder, but we use static loops below

        # Fully static loop over pool positions in rows
        for ph in tl.static_range(POH_C):
            for pw_start in range(0, POW_C, 16):
                pw_offs = pw_start + tl.arange(0, 16)
                pw_mask = pw_offs < POW_C
                # max over POOL_K x POOL_K window
                max_v = tl.full([16], -float('inf'), dtype=tl.float32)
                for kh in tl.static_range(POOL_K):
                    for kw in tl.static_range(POOL_K):
                        oh = ph * POOL_K + kh
                        ow = pw_offs * POOL_K + kw
                        idx = c_in_base + oh * OW + ow
                        v = tl.load(in_ptr + idx, mask=pw_mask, other=-float('inf'))
                        v = (v - mean) * a + bsh
                        max_v = tl.maximum(max_v, v)
                # clamp
                max_v = tl.minimum(tl.maximum(max_v, clamp_min), clamp_max)
                out_idx = c_out_base + ph * POW_C + pw_offs
                tl.store(out_ptr + out_idx, max_v, mask=pw_mask)


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

        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        scale_flat = self.scale.reshape(-1).contiguous()
        if scale_flat.numel() != OC:
            scale_flat = self.scale.expand(OC, 1, 1).reshape(-1).contiguous()

        # Conv output buffer
        conv_out = torch.empty((N, OC, OH * OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_PIX = 64
        grid_conv = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (OH * OW + BLOCK_PIX - 1) // BLOCK_PIX)
        conv_kernel[grid_conv](
            x, w, b, conv_out,
            N, IC, H, W, OC, OH, OW,
            KH=KH, KW=KW,
            IC_C=IC,
            BLOCK_OC=BLOCK_OC,
            BLOCK_PIX=BLOCK_PIX,
            num_warps=4,
            num_stages=2,
        )

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)
        out_view = out.view(N, OC, POH * POW)

        SPATIAL = OH * OW
        GROUP_SIZE = CH_PER_GROUP * SPATIAL
        BLOCK_SP = 512

        grid_gn = (N, self.num_groups)
        gn_scale_pool_clamp_kernel[grid_gn](
            conv_out, gn_w, gn_b, scale_flat, out_view,
            N, OC, OH, OW, POH, POW,
            GROUPS=self.num_groups,
            CH_PER_GROUP=CH_PER_GROUP,
            SPATIAL=SPATIAL,
            GROUP_SIZE=GROUP_SIZE,
            POOL_K=pool_k,
            POH_C=POH,
            POW_C=POW,
            BLOCK_SP=BLOCK_SP,
            eps=self.eps,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max,
            num_warps=4,
            num_stages=2,
        )

        return out