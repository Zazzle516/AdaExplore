import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_stats_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    N, OC, S, GROUPS,
    eps,
    CH_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # program per (n, group)
    n = tl.program_id(0)
    g = tl.program_id(1)
    
    sum_v = tl.zeros([BLOCK_S], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK_S], dtype=tl.float32)
    
    base = (n * OC + g * CH_PER_GROUP) * S
    
    for c in tl.static_range(CH_PER_GROUP):
        cbase = base + c * S
        for s_start in range(0, S, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            mask = s_offs < S
            v = tl.load(x_ptr + cbase + s_offs, mask=mask, other=0.0)
            sum_v += v
            sum_sq += v * v
    
    total = tl.sum(sum_v, axis=0)
    total_sq = tl.sum(sum_sq, axis=0)
    count = S * CH_PER_GROUP
    mean = total / count
    var = total_sq / count - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    
    tl.store(mean_ptr + n * GROUPS + g, mean)
    tl.store(rstd_ptr + n * GROUPS + g, rstd)


@triton.jit
def fused_epilogue_kernel(
    x_ptr, mean_ptr, rstd_ptr, gn_w_ptr, gn_b_ptr, scale_ptr, out_ptr,
    N, OC, OH, OW, POH, POW, GROUPS,
    clamp_min, clamp_max,
    POOL_K: tl.constexpr,
    CH_PER_GROUP: tl.constexpr,
    BLOCK_PIX: tl.constexpr,
):
    # program per (n, c, pool_tile)
    n = tl.program_id(0)
    c = tl.program_id(1)
    pt = tl.program_id(2)
    
    g = c // CH_PER_GROUP
    
    num_pool = POH * POW
    pool_offs = pt * BLOCK_PIX + tl.arange(0, BLOCK_PIX)
    pool_mask = pool_offs < num_pool
    
    poh = pool_offs // POW
    pow_ = pool_offs % POW
    
    mean = tl.load(mean_ptr + n * GROUPS + g)
    rstd = tl.load(rstd_ptr + n * GROUPS + g)
    gn_w = tl.load(gn_w_ptr + c)
    gn_b = tl.load(gn_b_ptr + c)
    sc = tl.load(scale_ptr + c)
    
    # combined affine: y = ((x - mean)*rstd*gn_w + gn_b) * sc
    # = x * (rstd*gn_w*sc) + (gn_b - mean*rstd*gn_w)*sc
    a = rstd * gn_w * sc
    b_ = (gn_b - mean * rstd * gn_w) * sc
    
    in_base = (n * OC + c) * OH * OW
    
    max_val = tl.full([BLOCK_PIX], -float('inf'), dtype=tl.float32)
    
    for pkh in tl.static_range(POOL_K):
        for pkw in tl.static_range(POOL_K):
            oh = poh * POOL_K + pkh
            ow = pow_ * POOL_K + pkw
            valid = pool_mask & (oh < OH) & (ow < OW)
            idx = in_base + oh * OW + ow
            v = tl.load(x_ptr + idx, mask=valid, other=-float('inf'))
            v = v * a + b_
            v = tl.where(valid, v, -float('inf'))
            max_val = tl.maximum(max_val, v)
    
    out_v = tl.minimum(tl.maximum(max_val, clamp_min), clamp_max)
    
    out_base = (n * OC + c) * num_pool + pool_offs
    tl.store(out_ptr + out_base, out_v, mask=pool_mask)


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
        # Conv via cuDNN (highly optimized)
        y = F.conv2d(x, self.conv.weight, self.conv.bias)
        N, OC, OH, OW = y.shape
        S = OH * OW
        pool_k = self.maxpool_kernel_size
        POH = OH // pool_k
        POW = OW // pool_k
        
        CH_PER_GROUP = OC // self.num_groups
        
        # Compute GN stats per (N, group)
        mean = torch.empty((N, self.num_groups), device=y.device, dtype=torch.float32)
        rstd = torch.empty((N, self.num_groups), device=y.device, dtype=torch.float32)
        
        BLOCK_S = 1024
        gn_stats_kernel[(N, self.num_groups)](
            y, mean, rstd,
            N, OC, S, self.num_groups,
            self.eps,
            CH_PER_GROUP=CH_PER_GROUP,
            BLOCK_S=BLOCK_S,
            num_warps=8,
            num_stages=3,
        )
        
        scale_flat = self.scale.reshape(-1).contiguous()
        if scale_flat.numel() != OC:
            scale_flat = self.scale.expand(OC, 1, 1).reshape(-1).contiguous()
        
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        
        out = torch.empty((N, OC, POH, POW), device=y.device, dtype=y.dtype)
        
        BLOCK_PIX = 128
        num_pool = POH * POW
        grid = (N, OC, (num_pool + BLOCK_PIX - 1) // BLOCK_PIX)
        
        fused_epilogue_kernel[grid](
            y, mean, rstd, gn_w, gn_b, scale_flat, out,
            N, OC, OH, OW, POH, POW, self.num_groups,
            self.clamp_min, self.clamp_max,
            POOL_K=pool_k,
            CH_PER_GROUP=CH_PER_GROUP,
            BLOCK_PIX=BLOCK_PIX,
            num_warps=4,
            num_stages=2,
        )
        
        return out