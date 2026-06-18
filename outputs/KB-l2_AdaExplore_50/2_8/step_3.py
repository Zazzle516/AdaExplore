import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_div_maxpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    PD, PH, PW,  # pooled output sizes
    inv_div,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    POOL_D: tl.constexpr, POOL_H: tl.constexpr, POOL_W: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # grid: (N * OC, PD * PH * PW)
    pid_nc = tl.program_id(0)
    pid_sp = tl.program_id(1)
    
    n = pid_nc // OC
    oc = pid_nc % OC
    
    pd = pid_sp // (PH * PW)
    rem = pid_sp % (PH * PW)
    ph = rem // PW
    pw = rem % PW
    
    # Starting output positions in conv output
    od_start = pd * POOL_D
    oh_start = ph * POOL_H
    ow_start = pw * POOL_W
    
    bias_val = tl.load(b_ptr + oc)
    
    max_val = -float('inf')
    
    for dd in tl.static_range(0, POOL_D):
        for hh in tl.static_range(0, POOL_H):
            for ww in tl.static_range(0, POOL_W):
                od = od_start + dd
                oh = oh_start + hh
                ow = ow_start + ww
                
                # Compute conv at (n, oc, od, oh, ow)
                acc = 0.0
                for kd in tl.static_range(0, KD):
                    for kh in tl.static_range(0, KH):
                        for kw in tl.static_range(0, KW):
                            id_ = od + kd
                            ih_ = oh + kh
                            iw_ = ow + kw
                            
                            ic_offs = tl.arange(0, BLOCK_IC)
                            ic_mask = ic_offs < IC
                            
                            # x: (N, IC, ID, IH, IW)
                            x_idx = ((n * IC + ic_offs) * ID + id_) * IH * IW + ih_ * IW + iw_
                            x_vals = tl.load(x_ptr + x_idx, mask=ic_mask, other=0.0)
                            
                            # w: (OC, IC, KD, KH, KW)
                            w_idx = ((oc * IC + ic_offs) * KD + kd) * KH * KW + kh * KW + kw
                            w_vals = tl.load(w_ptr + w_idx, mask=ic_mask, other=0.0)
                            
                            acc += tl.sum(x_vals * w_vals)
                
                acc = (acc + bias_val) * inv_div
                max_val = tl.maximum(max_val, acc)
    
    # Store to output (N, OC, PD, PH, PW)
    out_idx = ((n * OC + oc) * PD + pd) * PH * PW + ph * PW + pw
    tl.store(out_ptr + out_idx, max_val)


def conv3d_div_maxpool(x, weight, bias, divisor, pool_size):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    POOL_D, POOL_H, POOL_W = pool_size
    PD = OD // POOL_D
    PH = OH // POOL_H
    PW = OW // POOL_W
    
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    out = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=x.dtype)
    
    # BLOCK_IC must be >= IC, power of 2
    BLOCK_IC = 1
    while BLOCK_IC < IC:
        BLOCK_IC *= 2
    
    grid = (N * OC, PD * PH * PW)
    
    conv3d_div_maxpool_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        PD, PH, PW,
        1.0 / divisor,
        KD, KH, KW,
        POOL_D, POOL_H, POOL_W,
        BLOCK_IC,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.pool_size = pool_size
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous()
        # Fused conv + div + maxpool
        pooled = conv3d_div_maxpool(x, self.conv.weight, self.conv.bias, self.divisor, self.pool_size)
        # Global avg pool over D, H, W
        N, OC = pooled.shape[0], pooled.shape[1]
        avg = pooled.mean(dim=(2, 3, 4), keepdim=True)  # (N, OC, 1, 1, 1)
        avg = avg + self.bias
        out = avg.sum(dim=self.sum_dim)
        return out