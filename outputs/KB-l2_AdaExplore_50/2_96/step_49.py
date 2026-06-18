import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def convt3d_gather_kernel(
    x_ptr,    # (N, IC, ID, IH, IW)
    w_ptr,    # (IC, OC, KD, KH, KW)
    b_ptr,    # (OC,)
    out_ptr,  # (N, OC, OD, OH, OW)
    N, IC: tl.constexpr, ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr, OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N * OC, OD, ceil(OH*OW / BLOCK_HW))
    pid_noc = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    n = pid_noc // OC
    oc = pid_noc % OC
    od = pid_d
    
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < (OH * OW)
    oh = hw_offs // OW
    ow = hw_offs % OW
    
    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    
    # For each kernel position, find which input voxel contributes
    # out[od, oh, ow] += sum_ic, kd, kh, kw  x[id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where: id*SD - PD + kd = od  => id = (od + PD - kd) / SD if divisible
    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_ = id_num // SD
        d_valid = (id_num >= 0) & (id_num % SD == 0) & (id_ >= 0) & (id_ < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih = ih_num // SH
            h_valid = (ih_num >= 0) & (ih_num % SH == 0) & (ih >= 0) & (ih < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PW - kw
                iw = iw_num // SW
                w_valid = (iw_num >= 0) & (iw_num % SW == 0) & (iw >= 0) & (iw < IW)
                
                valid = d_valid & h_valid & w_valid & hw_mask
                
                # Sum over IC
                for ic in tl.static_range(0, IC):
                    x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
                    xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                    w_off = ((ic * OC + oc) * KD + kd) * KH * KW + kh * KW + kw
                    wv = tl.load(w_ptr + w_off)
                    acc += xv * wv
    
    # Add bias
    bv = tl.load(b_ptr + oc)
    acc += bv
    
    # Store to output
    out_off = ((n * OC + oc) * OD + od) * OH * OW + hw_offs
    tl.store(out_ptr + out_off, acc, mask=hw_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 512}, num_warps=4),
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 1024}, num_warps=8),
        triton.Config({'BLOCK': 2048}, num_warps=8),
    ],
    key=['D', 'H', 'W'],
)
@triton.jit
def fused_pool_avg_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    MK: tl.constexpr,
    scale: tl.constexpr,
    inv_count: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    total_pooled = PD * PH * PW
    base = (n * C + c) * D * H * W
    
    acc = 0.0
    for off in range(0, total_pooled, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < total_pooled
        
        pd = idx // (PH * PW)
        rem = idx % (PH * PW)
        ph = rem // PW
        pw = rem % PW
        
        d0 = pd * MK
        h0 = ph * MK
        w0 = pw * MK
        
        max_v = tl.full((BLOCK,), -float('inf'), dtype=tl.float32)
        for kd in tl.static_range(0, MK):
            for kh in tl.static_range(0, MK):
                for kw in tl.static_range(0, MK):
                    d_ = d0 + kd
                    h_ = h0 + kh
                    w_ = w0 + kw
                    in_bounds = (d_ < D) & (h_ < H) & (w_ < W) & mask
                    off_x = base + (d_ * H + h_) * W + w_
                    v = tl.load(x_ptr + off_x, mask=in_bounds, other=-float('inf'))
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
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding
        
        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW
        
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        b = self.conv_transpose.bias.contiguous()
        
        conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)
        
        BLOCK_HW = 128
        grid = (N * OC, OD, (OH * OW + BLOCK_HW - 1) // BLOCK_HW)
        convt3d_gather_kernel[grid](
            x, w, b, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
        )
        
        # Fused maxpool + global avg + scale + clamp
        MK = self.maxpool_kernel_size
        OD_P = OD // MK
        OH_P = OH // MK
        OW_P = OW // MK
        
        out = torch.empty((N, OC, 1, 1, 1), device=x.device, dtype=x.dtype)
        total_pooled = OD_P * OH_P * OW_P
        inv_count = 1.0 / float(total_pooled)
        
        fused_pool_avg_kernel[(N * OC,)](
            conv_out, out,
            N, OC, OD, OH, OW,
            OD_P, OH_P, OW_P,
            MK,
            float(self.scale),
            inv_count,
        )
        return out