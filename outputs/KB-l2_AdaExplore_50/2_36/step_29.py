import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_convtrans_min_sum_gelu_bias_kernel(
    x_ptr,        # [N, IC, IH, IW]
    w_ptr,        # [IC, OC, KH, KW]
    cb_ptr,       # conv bias [OC]
    bias_ptr,     # extra bias scalar
    out_ptr,      # [N, 1, 1, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # one program per (n, w_out)
    pid = tl.program_id(0)
    n = pid // OW
    w_out = pid % OW

    oc_off = tl.arange(0, BLOCK_OC)
    oc_mask = oc_off < OC

    # load conv bias for all OC
    cb = tl.load(cb_ptr + oc_off, mask=oc_mask, other=0.0)

    # We need to find which input w positions contribute to this w_out
    # For ConvTranspose2d: output[w_out] += sum over (iw, kw) where w_out = iw*STRIDE_W - PAD_W + kw
    # => iw = (w_out + PAD_W - kw) / STRIDE_W  (must be integer and in range)
    
    sum_over_h = tl.zeros([], dtype=tl.float32)

    # loop over h_out
    for h_out in range(0, OH):
        # accumulator for each output channel
        acc = tl.zeros([BLOCK_OC], dtype=tl.float32)
        
        # loop over kernel positions
        for kh in range(0, KH):
            # compute ih
            ih_num = h_out + PAD_H - kh
            ih = ih_num // STRIDE_H
            ih_valid = (ih_num >= 0) & (ih_num - ih * STRIDE_H == 0) & (ih >= 0) & (ih < IH)
            
            for kw in range(0, KW):
                iw_num = w_out + PAD_W - kw
                iw = iw_num // STRIDE_W
                iw_valid = (iw_num >= 0) & (iw_num - iw * STRIDE_W == 0) & (iw >= 0) & (iw < IW)
                
                valid = ih_valid & iw_valid
                
                # accumulate over IC
                # x[n, :, ih, iw] : [IC]
                # w[:, :, kh, kw] : [IC, OC]
                for ic_start in range(0, IC, BLOCK_IC):
                    ic_off = ic_start + tl.arange(0, BLOCK_IC)
                    ic_mask = ic_off < IC
                    
                    # load x[n, ic_off, ih, iw]
                    x_offs = n * IC * IH * IW + ic_off * IH * IW + ih * IW + iw
                    x_vals = tl.load(x_ptr + x_offs, mask=ic_mask & valid, other=0.0)  # [BLOCK_IC]
                    
                    # load w[ic_off, oc_off, kh, kw]
                    w_offs = ic_off[:, None] * OC * KH * KW + oc_off[None, :] * KH * KW + kh * KW + kw
                    w_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]
                    
                    # dot: sum over IC
                    acc += tl.sum(x_vals[:, None] * w_vals, axis=0)
        
        # add conv bias
        acc = acc + cb
        # mask invalid OC entries with +inf for min
        acc = tl.where(oc_mask, acc, float('inf'))
        # min over OC
        mn = tl.min(acc, axis=0)
        sum_over_h += mn
    
    # GELU
    g = 0.5 * sum_over_h * (1.0 + tl.erf(sum_over_h * 0.70710678118654752440))
    b = tl.load(bias_ptr)
    res = g + b
    
    out_ptr_off = n * OW + w_out
    tl.store(out_ptr + out_ptr_off, res)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        OC = self.out_channels
        STRIDE_H = STRIDE_W = self.stride
        PAD_H = PAD_W = self.padding
        # ConvTranspose2d output size
        OH = (IH - 1) * STRIDE_H - 2 * PAD_H + KH + self.output_padding
        OW = (IW - 1) * STRIDE_W - 2 * PAD_W + KW + self.output_padding

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        conv_bias = self.conv_transpose.bias.contiguous()

        out = torch.empty((N, 1, 1, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = triton.next_power_of_2(OC)
        BLOCK_IC = triton.next_power_of_2(IC)
        if BLOCK_IC > 64:
            BLOCK_IC = 64

        grid = (N * OW,)
        fused_convtrans_min_sum_gelu_bias_kernel[grid](
            x, weight, conv_bias, self.bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            STRIDE_H, STRIDE_W,
            PAD_H, PAD_W,
            BLOCK_OC=BLOCK_OC,
            BLOCK_IC=BLOCK_IC,
            num_warps=4,
        )
        return out