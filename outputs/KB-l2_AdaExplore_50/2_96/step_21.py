import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def convt3d_scatter_kernel(
    x_ptr,         # (N, IC, ID, IH, IW)
    w_ptr,         # (IC, OC, KD, KH, KW)
    b_ptr,         # (OC,)
    out_ptr,       # (N, OC, OD, OH, OW)
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, id, ih, iw): scatter input voxel * weight kernel into output
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    ih = pid_hw // IW
    iw = pid_hw % IW
    id_ = pid_d
    n = pid_n
    
    # output base coords
    od_base = id_ * SD - PD
    oh_base = ih * SH - PH
    ow_base = iw * SW - PW
    
    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC
    
    # load bias once if needed (we'll add it later or zero)
    
    # accumulate input value across IC for each oc and each kernel pos
    # For each (kd, kh, kw):
    for kd in tl.static_range(0, KD):
        od = od_base + kd
        d_valid = (od >= 0) & (od < OD)
        for kh in tl.static_range(0, KH):
            oh = oh_base + kh
            h_valid = (oh >= 0) & (oh < OH)
            for kw in tl.static_range(0, KW):
                ow = ow_base + kw
                w_valid = (ow >= 0) & (ow < OW)
                valid = d_valid & h_valid & w_valid
                
                # acc[oc] = sum_ic input[n, ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
                acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
                for ic in range(0, IC):
                    x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
                    xv = tl.load(x_ptr + x_off)
                    w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                    wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                    acc += xv * wv
                
                if valid:
                    out_off = ((n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
                    tl.atomic_add(out_ptr + out_off, acc, mask=oc_mask)


@triton.jit
def fused_pool_kernel(
    x_ptr,         # (N, C, D, H, W) - convT output
    out_ptr,       # (N, C)
    N, C, D, H, W,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,  # pooled dims
    MK: tl.constexpr,  # maxpool kernel
    scale: tl.constexpr,
    inv_count: tl.constexpr,
    BLOCK: tl.constexpr,
    C_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // (C // C_TILE)
    c_blk = pid % (C // C_TILE)
    c_base = c_blk * C_TILE
    
    total_pooled: tl.constexpr = PD * PH * PW
    HW = H * W
    DHW = D * HW
    
    # process C_TILE channels in this program
    for ci in tl.static_range(0, C_TILE):
        c = c_base + ci
        base = (n * C + c) * DHW
        
        acc = 0.0
        for off in tl.static_range(0, total_pooled, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            mask = idx < total_pooled
            
            pd = idx // (PH * PW)
            rem = idx % (PH * PW)
            ph = rem // PW
            pw = rem % PW
            
            d0 = pd * MK
            h0 = ph * MK
            w0 = pw * MK
            
            base_off = base + d0 * HW + h0 * W + w0
            
            # max over MK^3 window
            max_v = tl.full((BLOCK,), -float('inf'), dtype=tl.float32)
            for kd in tl.static_range(0, MK):
                for kh in tl.static_range(0, MK):
                    for kw in tl.static_range(0, MK):
                        off_x = base_off + kd * HW + kh * W + kw
                        v = tl.load(x_ptr + off_x, mask=mask, other=-float('inf'))
                        max_v = tl.maximum(max_v, v)
            
            max_v = tl.where(mask, max_v, 0.0)
            acc += tl.sum(max_v, axis=0)
        
        mean = acc * inv_count * scale
        mean = tl.minimum(tl.maximum(mean, 0.0), 1.0)
        tl.store(out_ptr + n * C + c, mean)


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
        # Use torch's convT3d (highly optimized cudnn) then fuse the tail
        x = self.conv_transpose(x)
        
        N, C, D, H, W = x.shape
        MK = self.maxpool_kernel_size
        PD = D // MK
        PH = H // MK
        PW = W // MK
        
        x = x.contiguous()
        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
        
        total_pooled = PD * PH * PW
        # choose block
        if total_pooled <= 128:
            BLOCK = 128
        elif total_pooled <= 256:
            BLOCK = 256
        elif total_pooled <= 512:
            BLOCK = 512
        else:
            BLOCK = 1024
        
        # tile channels
        if C % 4 == 0:
            C_TILE = 4
        elif C % 2 == 0:
            C_TILE = 2
        else:
            C_TILE = 1
        
        inv_count = 1.0 / float(total_pooled)
        grid = (N * (C // C_TILE),)
        fused_pool_kernel[grid](
            x, out,
            N, C, D, H, W,
            PD, PH, PW,
            MK,
            float(self.scale),
            inv_count,
            BLOCK=BLOCK,
            C_TILE=C_TILE,
            num_warps=8,
            num_stages=3,
        )
        return out