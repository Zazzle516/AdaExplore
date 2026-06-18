import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def convt3d_gather_kernel(
    x_ptr,         # (N, IC, ID, IH, IW)
    w_ptr,         # (IC, OC, KD, KH, KW)
    b_ptr,         # (OC,)
    out_ptr,       # (N, OC, PD, PH, PW)  -- pooled output (max over MK^3 window)
    N, IC, ID, IH, IW,
    OC,
    OD, OH, OW,
    PD, PH, PW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    MK: tl.constexpr,
    scale: tl.constexpr,
):
    # one program per (n, oc, pooled spatial voxel)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)
    
    pw = pid_sp % PW
    tmp = pid_sp // PW
    ph = tmp % PH
    pd = tmp // PH
    
    n = pid_n
    oc = pid_oc
    
    # max over MK^3 window in the convT output
    max_v = -float('inf')
    
    bias_v = tl.load(b_ptr + oc) * scale
    
    for mkd in tl.static_range(0, MK):
        od = pd * MK + mkd
        for mkh in tl.static_range(0, MK):
            oh = ph * MK + mkh
            for mkw in tl.static_range(0, MK):
                ow = pw * MK + mkw
                
                # compute output voxel value at (n, oc, od, oh, ow)
                acc = 0.0
                # for each kernel position, derive contributing input voxel
                for kd in tl.static_range(0, KD):
                    id_num = od + PAD_D - kd
                    id_ = id_num // SD
                    id_valid = (id_num >= 0) & ((id_num - id_ * SD) == 0) & (id_ >= 0) & (id_ < ID)
                    for kh in tl.static_range(0, KH):
                        ih_num = oh + PAD_H - kh
                        ih = ih_num // SH
                        ih_valid = (ih_num >= 0) & ((ih_num - ih * SH) == 0) & (ih >= 0) & (ih < IH)
                        for kw in tl.static_range(0, KW):
                            iw_num = ow + PAD_W - kw
                            iw = iw_num // SW
                            iw_valid = (iw_num >= 0) & ((iw_num - iw * SW) == 0) & (iw >= 0) & (iw < IW)
                            valid = id_valid & ih_valid & iw_valid
                            
                            # sum over IC
                            for ic in tl.static_range(0, 3):
                                x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
                                xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                                w_off = ((ic * OC + oc) * KD + kd) * KH * KW + kh * KW + kw
                                wv = tl.load(w_ptr + w_off)
                                acc += xv * wv
                
                val = acc * scale + bias_v
                max_v = tl.maximum(max_v, val)
    
    # store the max value for this pooled voxel
    out_off = ((n * OC + oc) * PD + pd) * PH * PW + ph * PW + pw
    tl.store(out_ptr + out_off, max_v)


@triton.jit
def avg_clamp_kernel(
    x_ptr,   # (N, C, PD, PH, PW)
    out_ptr, # (N, C)
    N, C, total,
    inv_count: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * total
    acc = 0.0
    for off in range(0, total, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < total
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        acc += tl.sum(v, axis=0)
    mean = acc * inv_count
    mean = tl.minimum(tl.maximum(mean, 0.0), 1.0)
    tl.store(out_ptr + pid, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        N = x.shape[0]
        IC = self.in_channels
        OC = self.out_channels
        ID, IH, IW = x.shape[2], x.shape[3], x.shape[4]
        
        K = self.kernel_size
        S = self.stride
        P = self.padding
        
        OD = (ID - 1) * S - 2 * P + K
        OH = (IH - 1) * S - 2 * P + K
        OW = (IW - 1) * S - 2 * P + K
        
        MK = self.maxpool_kernel_size
        PD = OD // MK
        PH = OH // MK
        PW = OW // MK
        
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()
        
        pooled = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=x.dtype)
        
        grid = (N, OC, PD * PH * PW)
        convt3d_gather_kernel[grid](
            x, weight, bias, pooled,
            N, IC, ID, IH, IW,
            OC,
            OD, OH, OW,
            PD, PH, PW,
            K, K, K,
            S, S, S,
            P, P, P,
            MK,
            float(self.scale),
            num_warps=4,
        )
        
        total = PD * PH * PW
        out = torch.empty((N, OC, 1, 1, 1), device=x.device, dtype=x.dtype)
        # pick BLOCK
        BLOCK = 1
        while BLOCK < total and BLOCK < 4096:
            BLOCK *= 2
        avg_clamp_kernel[(N * OC,)](
            pooled, out,
            N, OC, total,
            1.0 / float(total),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out