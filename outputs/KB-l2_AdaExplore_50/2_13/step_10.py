import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv_mean_kernel(
    x_ptr,        # (B, IC, D, H, W)
    w_ptr,        # (IC, OC, KD, KH, KW) - ConvTranspose3d weight layout
    cb_ptr,       # (OC,) - conv bias
    bias_ptr,     # (OC,) - extra bias
    out_ptr,      # (B, OC, H, W)
    B, IC, D, H, W,
    OC,
    scaling_factor,
    inv_D,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    IC_C: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    PAD: tl.constexpr,
):
    # Grid: (B, num_oc_tiles, num_hw_tiles)
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    HW = H * W
    
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    
    mask_oc = offs_oc < OC
    mask_hw = offs_hw < HW
    
    h_out = offs_hw // W  # [BLOCK_HW]
    w_out = offs_hw % W   # [BLOCK_HW]
    
    # Accumulate over D, KD, KH, KW, IC
    # ConvTranspose3d with stride=1, padding=PAD, kernel=K is equivalent to:
    # out[b, oc, d, h, w] = sum_{ic, kd, kh, kw} x[b, ic, d-kd+PAD, h-kh+PAD, w-kw+PAD] * W[ic, oc, kd, kh, kw]
    # We sum over d (depth-mean keepdim=True, then divide by D).
    # For each output spatial (h, w), accumulator over (oc) is:
    # acc[oc, hw] = sum_{d_out, ic, kd, kh, kw} x[b, ic, d_out-kd+PAD, ...] * W[ic, oc, kd, kh, kw]
    # 
    # Sum over d_out from 0..D-1 of x[b, ic, d_out-kd+PAD]: for each kd, this is
    # sum over d_in from (PAD-kd) to (D-1+PAD-kd) clipped to [0, D-1] of x[b, ic, d_in]
    # Equivalently: d_in ranges from max(0, PAD-kd) to min(D-1, D-1+PAD-kd)
    # For PAD=1, KD=3: kd=0: d_in in [1, D-1] (D-1 values), kd=1: [0, D-1] (D values), kd=2: [0, D-2] (D-1 values)
    # We need to compute x_sum[b, ic, kd] = sum over valid d_in of x[b, ic, d_in, h_in, w_in]
    # but h_in, w_in still depend on kh, kw, h_out, w_out.
    # 
    # Strategy: Accumulate properly without folding (per safety contract: must do full work).
    # We loop over kd, kh, kw, ic, d_out and accumulate.
    
    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)
    
    offs_ic = tl.arange(0, IC_C)
    mask_ic = offs_ic < IC
    
    # Loop over kernel positions
    for kd in tl.static_range(0, KD):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                # input spatial position for each output spatial
                h_in = h_out - kh + PAD  # [BLOCK_HW]
                w_in = w_out - kw + PAD  # [BLOCK_HW]
                spatial_valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask_hw
                
                # Load weight tile: W[ic, oc, kd, kh, kw] -> [IC_C, BLOCK_OC]
                # weight layout: (IC, OC, KD, KH, KW), stride: OC*KD*KH*KW, KD*KH*KW, KH*KW, KW, 1
                w_offs = (offs_ic[:, None] * (OC * KD * KH * KW)
                          + offs_oc[None, :] * (KD * KH * KW)
                          + kd * (KH * KW) + kh * KW + kw)
                w_mask = mask_ic[:, None] & mask_oc[None, :]
                w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [IC_C, BLOCK_OC]
                
                # Loop over d_out and accumulate sum of x over d_out
                # d_in = d_out - kd + PAD, valid when 0 <= d_in < D
                # d_out ranges: max(0, kd-PAD) to min(D-1, D-1+kd-PAD)
                d_out_lo = tl.maximum(0, kd - PAD)
                d_out_hi = tl.minimum(D - 1, D - 1 + kd - PAD)
                
                # Sum x over d_out range, for each ic and each (h_in, w_in)
                # x_sum[ic, hw] = sum_{d_out=lo..hi} x[b, ic, d_out-kd+PAD, h_in, w_in]
                # = sum_{d_in=lo-kd+PAD..hi-kd+PAD} x[b, ic, d_in, h_in, w_in]
                d_in_lo = d_out_lo - kd + PAD
                d_in_hi = d_out_hi - kd + PAD
                
                x_sum = tl.zeros((IC_C, BLOCK_HW), dtype=tl.float32)
                
                # Use a runtime loop over d_in
                for d_in in range(d_in_lo, d_in_hi + 1):
                    # Load x[b, ic, d_in, h_in, w_in] -> [IC_C, BLOCK_HW]
                    # x layout: (B, IC, D, H, W)
                    x_offs = (pid_b * (IC * D * H * W)
                              + offs_ic[:, None] * (D * H * W)
                              + d_in * (H * W)
                              + h_in[None, :] * W + w_in[None, :])
                    x_mask = mask_ic[:, None] & spatial_valid[None, :]
                    x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)
                    x_sum += x_vals
                
                # acc += w_vals.T @ x_sum, w_vals is [IC_C, BLOCK_OC], x_sum is [IC_C, BLOCK_HW]
                # acc[oc, hw] += sum_ic w_vals[ic, oc] * x_sum[ic, hw]
                acc += tl.dot(tl.trans(w_vals), x_sum)
    
    # Now acc holds sum over d_out (and ic, kd, kh, kw) -- divide by D for mean
    acc = acc * inv_D
    
    # Add conv bias and extra bias
    cb = tl.load(cb_ptr + offs_oc, mask=mask_oc, other=0.0)
    eb = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + (cb + eb)[:, None]
    
    # Store to (B, OC, H, W) output
    out_offs = (pid_b * (OC * HW)
                + offs_oc[:, None] * HW
                + offs_hw[None, :])
    out_mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


@triton.jit
def _softmax_tanh_scale_kernel(
    x_ptr,        # (B, C, H, W)
    out_ptr,      # (B, C, H, W)
    B, C, H, W,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    HW = H * W
    b = pid // HW
    hw = pid % HW
    
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    
    x_ptrs = x_ptr + b * C * HW + offs_c * HW + hw
    
    v = tl.load(x_ptrs, mask=mask_c, other=-float('inf')).to(tl.float32)
    
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask_c, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    
    e2 = tl.exp(2.0 * sm)
    t = (e2 - 1.0) / (e2 + 1.0)
    out = t * scaling_factor
    
    out_ptrs = out_ptr + b * C * HW + offs_c * HW + hw
    tl.store(out_ptrs, out, mask=mask_c)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scaling_factor = scaling_factor
        
        # Use the same parameter init as ConvTranspose3d
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
    
    def forward(self, x):
        B, IC, D, H, W = x.shape
        OC = self.out_channels
        K = self.kernel_size
        PAD = self.padding
        
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, K, K, K)
        cb = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        bias_flat = self.bias.view(-1).contiguous()
        
        # Output of conv+mean is (B, OC, H, W) since stride=1, padding=K//2
        reduced = torch.empty((B, OC, H, W), device=x.device, dtype=x.dtype)
        
        BLOCK_OC = 32
        BLOCK_HW = 64
        IC_C = 16  # IC = 16, fits exactly
        
        # Pad IC_C up to next power of 2 if needed
        ic_c = 1
        while ic_c < IC:
            ic_c *= 2
        IC_C = ic_c
        
        grid = (B, triton.cdiv(OC, BLOCK_OC), triton.cdiv(H * W, BLOCK_HW))
        
        _conv_mean_kernel[grid](
            x, weight, cb, bias_flat, reduced,
            B, IC, D, H, W, OC,
            float(self.scaling_factor),
            1.0 / float(D),
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            IC_C=IC_C,
            KD=K, KH=K, KW=K,
            PAD=PAD,
            num_warps=4,
            num_stages=2,
        )
        
        # Softmax + tanh + scale
        out = torch.empty_like(reduced)
        BLOCK_C = 1
        while BLOCK_C < OC:
            BLOCK_C *= 2
        
        grid2 = (B * H * W,)
        _softmax_tanh_scale_kernel[grid2](
            reduced, out,
            B, OC, H, W,
            float(self.scaling_factor),
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        
        return out.unsqueeze(2)