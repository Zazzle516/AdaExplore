import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    KD, KH, KW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # Each program: one (n, d_in, h_in, w_in) input position, all OC for one kd,kh,kw
    pid = tl.program_id(0)
    pid_kd = tl.program_id(1)
    pid_khw = tl.program_id(2)
    
    kh = pid_khw // KW
    kw = pid_khw % KW
    kd = pid_kd
    
    n = pid // (D_in * H_in * W_in)
    rem = pid % (D_in * H_in * W_in)
    d_in = rem // (H_in * W_in)
    rem2 = rem % (H_in * W_in)
    h_in = rem2 // W_in
    w_in = rem2 % W_in
    
    d_out = d_in * stride_d - pad_d + kd
    h_out = h_in * stride_h - pad_h + kh
    w_out = w_in * stride_w - pad_w + kw
    
    valid = (d_out >= 0) & (d_out < D_out) & (h_out >= 0) & (h_out < H_out) & (w_out >= 0) & (w_out < W_out)
    
    if valid:
        offs_oc = tl.arange(0, BLOCK_OC)
        mask_oc = offs_oc < OC
        
        # Accumulator for output values
        acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
        
        # Loop over IC
        for ic_start in range(0, IC, BLOCK_IC):
            offs_ic = ic_start + tl.arange(0, BLOCK_IC)
            mask_ic = offs_ic < IC
            
            # Load input: x[n, ic, d_in, h_in, w_in] for ic in offs_ic
            x_offs = n * (IC * D_in * H_in * W_in) + offs_ic * (D_in * H_in * W_in) + d_in * (H_in * W_in) + h_in * W_in + w_in
            x_vals = tl.load(x_ptr + x_offs, mask=mask_ic, other=0.0)  # [BLOCK_IC]
            
            # Load weight: w[ic, oc, kd, kh, kw] for ic in offs_ic, oc in offs_oc
            # weight shape: (IC, OC, KD, KH, KW)
            w_offs = (offs_ic[:, None] * (OC * KD * KH * KW) + 
                      offs_oc[None, :] * (KD * KH * KW) + 
                      kd * (KH * KW) + kh * KW + kw)
            w_mask = mask_ic[:, None] & mask_oc[None, :]
            w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]
            
            # Outer product reduction
            acc += tl.sum(x_vals[:, None] * w_vals, axis=0)
        
        # Atomic add to output: out[n, oc, d_out, h_out, w_out]
        out_offs = (n * (OC * D_out * H_out * W_out) + 
                    offs_oc * (D_out * H_out * W_out) + 
                    d_out * (H_out * W_out) + h_out * W_out + w_out)
        tl.atomic_add(out_ptr + out_offs, acc, mask=mask_oc)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, D_in, H_in, W_in = x.shape
    _, OC, KD, KH, KW = weight.shape
    
    sd, sh, sw = stride, stride, stride
    pd, ph, pw = padding, padding, padding
    
    D_out = (D_in - 1) * sd - 2 * pd + KD
    H_out = (H_in - 1) * sh - 2 * ph + KH
    W_out = (W_in - 1) * sw - 2 * pw + KW
    
    if bias is not None:
        out = bias.view(1, OC, 1, 1, 1).expand(N, OC, D_out, H_out, W_out).contiguous()
    else:
        out = torch.zeros(N, OC, D_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    BLOCK_OC = triton.next_power_of_2(OC)
    BLOCK_IC = min(triton.next_power_of_2(IC), 32)
    
    grid = (N * D_in * H_in * W_in, KD, KH * KW)
    
    conv_transpose3d_scatter_kernel[grid](
        x, weight, out,
        N, IC, OC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        KD, KH, KW,
        sd, sh, sw,
        pd, ph, pw,
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC,
        num_warps=4,
        num_stages=2,
    )
    
    return out


@triton.jit
def layernorm_gelu_scale_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    M, C,
    eps, scaling_factor,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    x = tl.load(x_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
    
    mean = tl.sum(x, axis=0) / C
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    
    y = xm * rstd * w + b
    inv_sqrt2 = 0.70710678118654752440
    g = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    g = g * scaling_factor
    
    tl.store(out_ptr + row * C + offs, g, mask=mask)


def fused_ln_gelu_scale(x, weight, bias, eps, scaling_factor):
    # x shape after permute: (N, D, H, W, C) where C = out_channels is last and contiguous
    assert x.is_contiguous()
    shape = x.shape
    C = shape[-1]
    M = x.numel() // C
    out = torch.empty_like(x)
    
    BLOCK_C = triton.next_power_of_2(C)
    
    layernorm_gelu_scale_kernel[(M,)](
        x, out, weight, bias,
        M, C,
        eps, scaling_factor,
        BLOCK_C=BLOCK_C,
        num_warps=2,
        num_stages=2,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.eps = eps
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias if self.use_bias else None
        
        y = conv_transpose3d_triton(x, weight, bias, self.stride, self.padding)
        # Permute to put channel last for layernorm
        # y shape: (N, OC, D, H, W) -> (N, D, H, W, OC)
        y = y.permute(0, 2, 3, 4, 1).contiguous()
        y = fused_ln_gelu_scale(y, self.layer_norm.weight, self.layer_norm.bias, self.eps, self.scaling_factor)
        # Permute back
        y = y.permute(0, 4, 1, 2, 3).contiguous()
        return y