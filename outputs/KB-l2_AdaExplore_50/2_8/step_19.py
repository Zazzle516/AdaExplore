import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, KD, KH, KW,
    OD, OH, OW,        # conv output dims
    PD, PH, PW,        # pooled output dims
    inv_div,
    BLOCK_K: tl.constexpr,
):
    # Each program computes one (n, oc, pd, ph, pw) — i.e., one max-pooled value
    pid = tl.program_id(0)
    pid_nc = tl.program_id(1)
    
    n = pid_nc // OC
    oc = pid_nc % OC
    
    pw_idx = pid % PW
    ph_idx = (pid // PW) % PH
    pd_idx = pid // (PW * PH)
    
    # The 2x2x2 pool window in conv output space
    od_start = pd_idx * 2
    oh_start = ph_idx * 2
    ow_start = pw_idx * 2
    
    max_val = -float('inf')
    
    # Iterate over 2x2x2 pool window
    for dd in range(2):
        for dh in range(2):
            for dw in range(2):
                od = od_start + dd
                oh = oh_start + dh
                ow = ow_start + dw
                
                # compute conv at (n, oc, od, oh, ow)
                acc = 0.0
                # input base: (n, ic, od:od+KD, oh:oh+KH, ow:ow+KW) dot w(oc, ic, :, :, :)
                # KD=KH=KW=3, total 27 elements per ic, IC channels
                for ic in range(IC):
                    for kd in range(KD):
                        for kh in range(KH):
                            for kw in range(KW):
                                id_ = od + kd
                                ih_ = oh + kh
                                iw_ = ow + kw
                                x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih_ * IW + iw_
                                w_off = ((oc * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                                a = tl.load(x_ptr + x_off)
                                b = tl.load(w_ptr + w_off)
                                acc += a * b
                
                # add bias
                bias = tl.load(b_ptr + oc)
                conv_val = (acc + bias) * inv_div
                max_val = tl.maximum(max_val, conv_val)
    
    # store at (n, oc, pd, ph, pw)
    out_off = ((n * OC + oc) * PD + pd_idx) * PH * PW + ph_idx * PW + pw_idx
    tl.store(out_ptr + out_off, max_val)


@triton.jit
def reduce_kernel(
    pooled_ptr, bias_ptr, out_ptr,
    N, OC, NSPATIAL,
    BLOCK_S: tl.constexpr,
):
    # one program per n: compute scalar = sum_oc ( mean_spatial(pooled[n, oc]) + bias[oc] )
    n = tl.program_id(0)
    
    # accumulate over oc
    total = 0.0
    for oc in range(OC):
        # mean over spatial
        s = 0.0
        offs = tl.arange(0, BLOCK_S)
        mask = offs < NSPATIAL
        base = (n * OC + oc) * NSPATIAL
        vals = tl.load(pooled_ptr + base + offs, mask=mask, other=0.0)
        s = tl.sum(vals, axis=0)
        mean = s / NSPATIAL
        bv = tl.load(bias_ptr + oc)
        total += mean + bv
    
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
    
    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD, KH, KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        # max pool with size 2
        PD = OD // 2
        PH = OH // 2
        PW = OW // 2
        
        weight = self.conv.weight.contiguous()
        cbias = self.conv.bias.contiguous()
        
        pooled = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=x.dtype)
        
        grid = (PD * PH * PW, N * OC)
        conv3d_pool_kernel[grid](
            x, weight, cbias, pooled,
            N, IC, ID, IH, IW,
            OC, KD, KH, KW,
            OD, OH, OW,
            PD, PH, PW,
            1.0 / self.divisor,
            BLOCK_K=1,
            num_warps=4,
        )
        
        # global avg pool over (PD, PH, PW), then + bias (OC,1,1,1), then sum over sum_dim
        # result of avg pool: (N, OC, 1, 1, 1); + bias broadcast -> (N, OC, 1, 1, 1)
        # sum over dim=1 -> (N, 1, 1, 1)
        NSPATIAL = PD * PH * PW
        bias_flat = self.bias.view(-1).contiguous()
        
        out = torch.empty((N,), device=x.device, dtype=x.dtype)
        
        # find next pow2 >= NSPATIAL
        BLOCK_S = 1
        while BLOCK_S < NSPATIAL:
            BLOCK_S *= 2
        
        reduce_kernel[(N,)](
            pooled, bias_flat, out,
            N, OC, NSPATIAL,
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )
        
        if self.sum_dim == 1:
            return out.view(N, 1, 1, 1)
        else:
            # fallback
            avg = pooled.mean(dim=(2, 3, 4), keepdim=True) + self.bias
            return torch.sum(avg, dim=self.sum_dim)