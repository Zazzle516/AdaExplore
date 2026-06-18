import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_div_pool_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,  # conv output dims
    PD, PH, PW,      # pooled dims (after maxpool)
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    POOL_D: tl.constexpr, POOL_H: tl.constexpr, POOL_W: tl.constexpr,
    inv_divisor,
    inv_pool_vol,
    BLOCK_OC: tl.constexpr,
):
    # one program per batch n
    n = tl.program_id(0)
    
    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC
    
    # accumulator for global avg pool sum: per (n, oc)
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)
    
    # Iterate over all pooled output positions
    total_pool = PD * PH * PW
    
    for p_idx in range(total_pool):
        pd = p_idx // (PH * PW)
        rem = p_idx % (PH * PW)
        ph = rem // PW
        pw = rem % PW
        
        # max pool window in conv output space
        od_start = pd * POOL_D
        oh_start = ph * POOL_H
        ow_start = pw * POOL_W
        
        # max over pool window
        max_vals = tl.full([BLOCK_OC], -float('inf'), dtype=tl.float32)
        
        for kd in range(POOL_D):
            for kh in range(POOL_H):
                for kw in range(POOL_W):
                    od = od_start + kd
                    oh = oh_start + kh
                    ow = ow_start + kw
                    
                    # compute conv output at (n, oc, od, oh, ow) for all oc in block
                    conv_val = tl.zeros([BLOCK_OC], dtype=tl.float32)
                    
                    for ic in range(IC):
                        for kkd in tl.static_range(KD):
                            for kkh in tl.static_range(KH):
                                for kkw in tl.static_range(KW):
                                    id_ = od + kkd
                                    ih = oh + kkh
                                    iw = ow + kkw
                                    
                                    x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
                                    x_val = tl.load(x_ptr + x_off)
                                    
                                    # weight: [OC, IC, KD, KH, KW]
                                    w_off = oc_offs * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kkd * (KH * KW) + kkh * KW + kkw
                                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                    
                                    conv_val += x_val * w_val
                    
                    # add conv bias
                    cb = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
                    conv_val += cb
                    
                    # divide
                    conv_val = conv_val * inv_divisor
                    
                    max_vals = tl.maximum(max_vals, conv_val)
        
        acc += max_vals
    
    # global avg pool: divide by total_pool
    acc = acc * inv_pool_vol
    
    # add bias (per channel)
    bias_val = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias_val
    
    # sum over OC dim -> scalar per n
    # but we need to mask
    acc_masked = tl.where(oc_mask, acc, 0.0)
    total = tl.sum(acc_masked, axis=0)
    
    # output shape after sum_dim=1: (N, 1, 1, 1) since avgpool gives (N,C,1,1,1) then bias added then sum over dim 1
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
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD, KH, KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL_D, POOL_H, POOL_W = self.pool_size
        PD = OD // POOL_D
        PH = OH // POOL_H
        PW = OW // POOL_W
        
        # Output: sum over channel dim of (N, C, 1, 1, 1) + bias -> (N, 1, 1, 1)
        out = torch.empty(N, 1, 1, 1, device=x.device, dtype=x.dtype)
        
        BLOCK_OC = triton.next_power_of_2(OC)
        if BLOCK_OC < 16:
            BLOCK_OC = 16
        
        bias_flat = self.bias.view(-1).contiguous()
        
        grid = (N,)
        fused_conv_div_pool_kernel[grid](
            x, self.conv.weight, self.conv.bias, bias_flat, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            PD, PH, PW,
            KD, KH, KW,
            POOL_D, POOL_H, POOL_W,
            1.0 / self.divisor,
            1.0 / (PD * PH * PW),
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
        )
        
        return out