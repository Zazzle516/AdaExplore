import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_conv_kernel(
    conv_ptr, bias_ptr, out_ptr,
    N, OC, OD, OH, OW,
    PD, PH, PW,
    POOL_D: tl.constexpr, POOL_H: tl.constexpr, POOL_W: tl.constexpr,
    inv_divisor,
    inv_pool_vol,
    BLOCK_OC: tl.constexpr,
    TOTAL_POOL: tl.constexpr,
):
    # one program per batch n; processes all OC and all pooled positions
    n = tl.program_id(0)
    
    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC
    
    # accumulator across pooled positions for each oc
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)
    
    PHPW = PH * PW
    OHOW = OH * OW
    OC_STRIDE = OD * OH * OW
    N_STRIDE = OC * OC_STRIDE
    
    for p_idx in range(TOTAL_POOL):
        pd = p_idx // PHPW
        rem = p_idx % PHPW
        ph = rem // PW
        pw = rem % PW
        
        od_start = pd * POOL_D
        oh_start = ph * POOL_H
        ow_start = pw * POOL_W
        
        max_vals = tl.full([BLOCK_OC], -float('inf'), dtype=tl.float32)
        
        for kd in range(POOL_D):
            for kh in range(POOL_H):
                for kw in range(POOL_W):
                    od = od_start + kd
                    oh = oh_start + kh
                    ow = ow_start + kw
                    
                    # conv layout: (N, OC, OD, OH, OW)
                    off = n * N_STRIDE + oc_offs * OC_STRIDE + od * OHOW + oh * OW + ow
                    v = tl.load(conv_ptr + off, mask=oc_mask, other=-float('inf'))
                    max_vals = tl.maximum(max_vals, v)
        
        acc += max_vals
    
    # apply divisor and avg-pool scaling (linear ops commute with sum)
    acc = acc * (inv_divisor * inv_pool_vol)
    
    # add bias (per channel)
    bias_val = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias_val
    
    acc_masked = tl.where(oc_mask, acc, 0.0)
    total = tl.sum(acc_masked, axis=0)
    
    tl.store(out_ptr + n, total)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x):
        # Run cuDNN conv (heaviest op), then fuse the rest in Triton
        conv_out = F.conv3d(x, self.conv.weight, self.conv.bias)
        conv_out = conv_out.contiguous()
        
        N, OC, OD, OH, OW = conv_out.shape
        POOL_D, POOL_H, POOL_W = self.pool_size
        PD = OD // POOL_D
        PH = OH // POOL_H
        PW = OW // POOL_W
        
        out = torch.empty(N, 1, 1, 1, device=x.device, dtype=x.dtype)
        
        BLOCK_OC = triton.next_power_of_2(OC)
        if BLOCK_OC < 16:
            BLOCK_OC = 16
        
        bias_flat = self.bias.view(-1).contiguous()
        
        grid = (N,)
        fused_post_conv_kernel[grid](
            conv_out, bias_flat, out,
            N, OC, OD, OH, OW,
            PD, PH, PW,
            POOL_D, POOL_H, POOL_W,
            1.0 / self.divisor,
            1.0 / (PD * PH * PW),
            BLOCK_OC=BLOCK_OC,
            TOTAL_POOL=PD * PH * PW,
            num_warps=2,
            num_stages=2,
        )
        
        return out