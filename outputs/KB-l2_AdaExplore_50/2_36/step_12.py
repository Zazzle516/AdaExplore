import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H_in, W_in,
    OC, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # one program per (n, oh, ow) ; computes BLOCK_OC channels
    pid = tl.program_id(0)
    pid_oc = tl.program_id(1)
    
    HW_out = H_out * W_out
    n = pid // HW_out
    rem = pid % HW_out
    oh = rem // W_out
    ow = rem % W_out
    
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < OC
    
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)
    
    # ConvTranspose: out[n,oc,oh,ow] = sum_{ic, kh, kw} x[n,ic,ih,iw] * w[ic,oc,kh,kw]
    # where ih*stride - pad + kh = oh  =>  ih = (oh + pad - kh) / stride
    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        ih_valid = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < H_in)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < W_in)
            valid = ih_valid & iw_valid
            
            # iterate over IC in blocks
            for ic_start in range(0, IC, BLOCK_IC):
                offs_ic = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = offs_ic < IC
                
                # x[n, ic, ih, iw]
                x_offs = n * IC * H_in * W_in + offs_ic * H_in * W_in + ih * W_in + iw
                x_vals = tl.load(x_ptr + x_offs, mask=ic_mask & valid, other=0.0)  # [BLOCK_IC]
                
                # w[ic, oc, kh, kw]
                w_offs = offs_ic[:, None] * OC * KH * KW + offs_oc[None, :] * KH * KW + kh * KW + kw
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]
                
                acc += tl.sum(x_vals[:, None] * w_vals, axis=0)
    
    # add bias
    b = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc += b
    
    # store
    out_offs = n * OC * HW_out + offs_oc * HW_out + oh * W_out + ow
    tl.store(out_ptr + out_offs, acc, mask=oc_mask)


@triton.jit
def fused_min_sum_gelu_bias_kernel(
    x_ptr, out_ptr, bias_ptr,
    N, C, H, W,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # one program per (n, w) ; loops over h, computing min over C, then sum
    pid = tl.program_id(0)
    n = pid // W
    w = pid % W
    
    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    
    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    
    HW = H * W
    # x[n, c, h, w] for all c, h
    # base for (n, 0, 0, w) is n*C*HW + w
    base = n * C * HW + w
    # offsets: c * HW + h * W
    offs = offs_c[:, None] * HW + offs_h[None, :] * W  # [BLOCK_C, BLOCK_H]
    mask = c_mask[:, None] & h_mask[None, :]
    
    x = tl.load(x_ptr + base + offs, mask=mask, other=float('inf'))
    # min over C
    m = tl.min(x, axis=0)  # [BLOCK_H]
    # mask out invalid h positions before sum
    m = tl.where(h_mask, m, 0.0)
    s = tl.sum(m, axis=0)  # scalar
    
    # GELU exact
    inv_sqrt2 = 0.70710678118654752440
    g = 0.5 * s * (1.0 + tl.erf(s * inv_sqrt2))
    
    b = tl.load(bias_ptr)
    out = g + b
    tl.store(out_ptr + n * W + w, out)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


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
        N, IC, H_in, W_in = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        H_out = (H_in - 1) * self.stride - 2 * self.padding + KH + self.output_padding
        W_out = (W_in - 1) * self.stride - 2 * self.padding + KW + self.output_padding
        
        # Step 1: ConvTranspose2d via custom kernel
        ct_out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)
        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        conv_bias = self.conv_transpose.bias.contiguous()  # [OC]
        
        BLOCK_OC = 128
        BLOCK_IC = 16
        grid = (N * H_out * W_out, triton.cdiv(OC, BLOCK_OC))
        conv_transpose_kernel[grid](
            x, weight, conv_bias, ct_out,
            N, IC, H_in, W_in,
            OC, H_out, W_out,
            KH, KW,
            self.stride, self.padding,
            BLOCK_OC=BLOCK_OC,
            BLOCK_IC=BLOCK_IC,
            num_warps=4,
        )
        
        # Step 2: fused min-sum-gelu-bias
        out = torch.empty((N, 1, 1, W_out), device=x.device, dtype=x.dtype)
        BLOCK_C = _next_pow2(OC)
        BLOCK_H = _next_pow2(H_out)
        grid2 = (N * W_out,)
        fused_min_sum_gelu_bias_kernel[grid2](
            ct_out, out, self.bias,
            N, OC, H_out, W_out,
            BLOCK_C=BLOCK_C,
            BLOCK_H=BLOCK_H,
            num_warps=8,
        )
        
        return out