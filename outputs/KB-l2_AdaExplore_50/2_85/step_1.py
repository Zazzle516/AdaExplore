import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_gn_scale_maxpool_clamp_kernel(
    x_ptr, w_ptr, b_ptr, gn_w_ptr, gn_b_ptr, scale_ptr, out_ptr,
    N, IC, H, W, OC, KH, KW, OH, OW, POH, POW,
    GROUPS, CH_PER_GROUP,
    clamp_min, clamp_max,
    eps,
    pool_k: tl.constexpr,
    BLOCK_PIX: tl.constexpr,
    IC_C: tl.constexpr,
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
    CH_PER_GROUP_C: tl.constexpr,
):
    # program ids: (n, group, pool_tile)
    n = tl.program_id(0)
    g = tl.program_id(1)
    pt = tl.program_id(2)

    num_pool = POH * POW
    pool_offs = pt * BLOCK_PIX + tl.arange(0, BLOCK_PIX)
    pool_mask = pool_offs < num_pool

    poh = pool_offs // POW
    pow_ = pool_offs % POW

    # We need to compute conv output for the CH_PER_GROUP_C channels in this group,
    # over the pool_k x pool_k window for each pool position.
    # Then group norm stats need to be computed over ALL spatial positions and channels in group.
    # 
    # Strategy: compute conv output for full group spatial map first, store in shared/registers via two passes.
    # Pass 1: compute conv output for entire group, accumulate sum and sum_sq for GN stats.
    # Pass 2: recompute conv output, apply GN+scale, do maxpool over pool_k x pool_k window, clamp.
    #
    # Since output is large, we do pass 1 over the full OH*OW for this group to get stats.
    
    # === Pass 1: compute GN stats ===
    # iterate over all output pixels for this group
    sum_val = tl.zeros([CH_PER_GROUP_C], dtype=tl.float32)
    sum_sq = tl.zeros([CH_PER_GROUP_C], dtype=tl.float32)
    
    total_pix = OH * OW
    PIX_BLOCK: tl.constexpr = 64
    
    c_offs = g * CH_PER_GROUP_C + tl.arange(0, CH_PER_GROUP_C)  # [CH]
    
    for pix_start in range(0, total_pix, PIX_BLOCK):
        pix_ids = pix_start + tl.arange(0, PIX_BLOCK)
        pix_m = pix_ids < total_pix
        oh = pix_ids // OW
        ow = pix_ids % OW
        
        acc = tl.zeros([PIX_BLOCK, CH_PER_GROUP_C], dtype=tl.float32)
        
        for ic in tl.static_range(IC_C):
            for kh in tl.static_range(KH_C):
                for kw in tl.static_range(KW_C):
                    ih = oh + kh  # [PIX_BLOCK]
                    iw = ow + kw
                    x_idx = ((n * IC + ic) * H + ih) * W + iw
                    x_val = tl.load(x_ptr + x_idx, mask=pix_m, other=0.0)  # [PIX_BLOCK]
                    # weight: [OC, IC, KH, KW]
                    w_idx = ((c_offs * IC + ic) * KH_C + kh) * KW_C + kw  # [CH]
                    w_val = tl.load(w_ptr + w_idx)  # [CH]
                    acc += x_val[:, None] * w_val[None, :]
        
        # add bias
        b_val = tl.load(b_ptr + c_offs)  # [CH]
        acc = acc + b_val[None, :]
        
        # mask out invalid
        acc_masked = tl.where(pix_m[:, None], acc, 0.0)
        sum_val += tl.sum(acc_masked, axis=0)
        sum_sq += tl.sum(acc_masked * acc_masked, axis=0)
    
    # compute mean/var over the group (all channels and spatial)
    total_count = total_pix * CH_PER_GROUP_C
    total_sum = tl.sum(sum_val, axis=0)
    total_sumsq = tl.sum(sum_sq, axis=0)
    mean = total_sum / total_count
    var = total_sumsq / total_count - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # load gn weight, bias, scale for these channels
    gn_w = tl.load(gn_w_ptr + c_offs)  # [CH]
    gn_b = tl.load(gn_b_ptr + c_offs)
    sc = tl.load(scale_ptr + c_offs)
    
    # === Pass 2: for each pool position in our block, compute maxpool ===
    # For each pool position (poh, pow_), we need conv output over pool_k x pool_k window
    # at oh in [poh*pool_k, poh*pool_k + pool_k), ow in [pow_*pool_k, pow_*pool_k + pool_k)
    # for each of CH_PER_GROUP_C channels.
    
    # accumulator for max over each channel, each pool position
    max_val = tl.full([BLOCK_PIX, CH_PER_GROUP_C], -float('inf'), dtype=tl.float32)
    
    for pkh in tl.static_range(pool_k):
        for pkw in tl.static_range(pool_k):
            oh = poh * pool_k + pkh  # [BLOCK_PIX]
            ow = pow_ * pool_k + pkw
            
            valid = pool_mask & (oh < OH) & (ow < OW)
            
            acc = tl.zeros([BLOCK_PIX, CH_PER_GROUP_C], dtype=tl.float32)
            for ic in tl.static_range(IC_C):
                for kh in tl.static_range(KH_C):
                    for kw in tl.static_range(KW_C):
                        ih = oh + kh
                        iw = ow + kw
                        x_idx = ((n * IC + ic) * H + ih) * W + iw
                        x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
                        w_idx = ((c_offs * IC + ic) * KH_C + kh) * KW_C + kw
                        w_val = tl.load(w_ptr + w_idx)
                        acc += x_val[:, None] * w_val[None, :]
            
            b_val = tl.load(b_ptr + c_offs)
            acc = acc + b_val[None, :]
            
            # apply group norm: (acc - mean) * rstd * gn_w + gn_b
            normed = (acc - mean) * rstd * gn_w[None, :] + gn_b[None, :]
            # apply scale
            scaled = normed * sc[None, :]
            
            scaled = tl.where(valid[:, None], scaled, -float('inf'))
            max_val = tl.maximum(max_val, scaled)
    
    # clamp
    out_v = tl.minimum(tl.maximum(max_val, clamp_min), clamp_max)
    
    # store: output shape [N, OC, POH, POW]
    # For each (channel, pool position):
    out_base = (n * OC + c_offs[None, :]) * (POH * POW) + pool_offs[:, None]
    store_mask = pool_mask[:, None] & tl.full([1, CH_PER_GROUP_C], True, dtype=tl.int1)
    tl.store(out_ptr + out_base, out_v, mask=store_mask)


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
        
        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)
        
        scale_flat = self.scale.view(-1).contiguous()
        if scale_flat.numel() != OC:
            # broadcast - for scale_shape (OC,1,1), view(-1) gives OC
            scale_flat = self.scale.expand(OC, 1, 1).reshape(-1).contiguous()
        
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        
        BLOCK_PIX = 32
        num_pool = POH * POW
        grid = (N, self.num_groups, (num_pool + BLOCK_PIX - 1) // BLOCK_PIX)
        
        conv_gn_scale_maxpool_clamp_kernel[grid](
            x, w, b, gn_w, gn_b, scale_flat, out,
            N, IC, H, W, OC, KH, KW, OH, OW, POH, POW,
            self.num_groups, CH_PER_GROUP,
            self.clamp_min, self.clamp_max,
            self.eps,
            pool_k=pool_k,
            BLOCK_PIX=BLOCK_PIX,
            IC_C=IC,
            KH_C=KH,
            KW_C=KW,
            CH_PER_GROUP_C=CH_PER_GROUP,
            num_warps=4,
            num_stages=2,
        )
        
        return out