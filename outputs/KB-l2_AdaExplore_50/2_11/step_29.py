import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_bn_tanh_maxpool_gn_kernel(
    x_ptr,           # input after conv_transpose: (N, C, H, W)
    out_ptr,         # output: (N, C, H//2, W//2)
    bn_scale_ptr,    # per-channel scale = bn_weight / sqrt(running_var + eps)
    bn_bias_ptr,     # per-channel bias = bn_bias - running_mean * bn_scale
    gn_weight_ptr,   # (C,)
    gn_bias_ptr,     # (C,)
    N, C, H, W,
    H_out, W_out,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    SPATIAL_OUT: tl.constexpr,  # H_out * W_out
    GROUP_SIZE: tl.constexpr,    # CHANNELS_PER_GROUP * SPATIAL_OUT
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    # Compute pooled+activated values for this (n, g) block, accumulating mean/var
    # We'll do it in two passes by recomputing — simpler: store to output then reload.
    # Pass 1: compute fused bn+tanh+maxpool, write to output, accumulate sum and sumsq
    
    offs = tl.arange(0, BLOCK)
    
    sum_val = 0.0
    sumsq_val = 0.0
    
    # Iterate over the group's elements in chunks
    # Group has CHANNELS_PER_GROUP channels, each with SPATIAL_OUT output elements
    num_chunks = (GROUP_SIZE + BLOCK - 1) // BLOCK
    
    for chunk in range(num_chunks):
        idx = chunk * BLOCK + offs
        mask = idx < GROUP_SIZE
        
        # Decompose idx -> (c_in_group, sp_out)
        c_in_group = idx // SPATIAL_OUT
        sp_out = idx % SPATIAL_OUT
        c = g * CHANNELS_PER_GROUP + c_in_group
        
        h_out = sp_out // W_out
        w_out = sp_out % W_out
        
        # 2x2 max pool: source positions
        h_in_base = h_out * 2
        w_in_base = w_out * 2
        
        # Load BN scale/bias for the channel
        scale = tl.load(bn_scale_ptr + c, mask=mask, other=0.0)
        bias = tl.load(bn_bias_ptr + c, mask=mask, other=0.0)
        
        # Compute base offset into x: n*C*H*W + c*H*W
        base = n * C * H * W + c * H * W
        
        # 4 positions
        v00 = tl.load(x_ptr + base + h_in_base * W + w_in_base, mask=mask, other=-1e30)
        v01 = tl.load(x_ptr + base + h_in_base * W + (w_in_base + 1), mask=mask, other=-1e30)
        v10 = tl.load(x_ptr + base + (h_in_base + 1) * W + w_in_base, mask=mask, other=-1e30)
        v11 = tl.load(x_ptr + base + (h_in_base + 1) * W + (w_in_base + 1), mask=mask, other=-1e30)
        
        # Apply BN + tanh fused
        t00 = tl.extra.cuda.libdevice.tanh(v00 * scale + bias)
        t01 = tl.extra.cuda.libdevice.tanh(v01 * scale + bias)
        t10 = tl.extra.cuda.libdevice.tanh(v10 * scale + bias)
        t11 = tl.extra.cuda.libdevice.tanh(v11 * scale + bias)
        
        # Max pool
        m0 = tl.maximum(t00, t01)
        m1 = tl.maximum(t10, t11)
        pooled = tl.maximum(m0, m1)
        
        pooled = tl.where(mask, pooled, 0.0)
        
        # Store to output (intermediate location)
        out_base = n * C * SPATIAL_OUT + c * SPATIAL_OUT + sp_out
        tl.store(out_ptr + out_base, pooled, mask=mask)
        
        sum_val += tl.sum(pooled)
        sumsq_val += tl.sum(pooled * pooled)
    
    # Compute mean / var for the group
    mean = sum_val / GROUP_SIZE
    var = sumsq_val / GROUP_SIZE - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)
    
    # Pass 2: apply group norm with affine
    for chunk in range(num_chunks):
        idx = chunk * BLOCK + offs
        mask = idx < GROUP_SIZE
        
        c_in_group = idx // SPATIAL_OUT
        sp_out = idx % SPATIAL_OUT
        c = g * CHANNELS_PER_GROUP + c_in_group
        
        gn_w = tl.load(gn_weight_ptr + c, mask=mask, other=0.0)
        gn_b = tl.load(gn_bias_ptr + c, mask=mask, other=0.0)
        
        out_base = n * C * SPATIAL_OUT + c * SPATIAL_OUT + sp_out
        val = tl.load(out_ptr + out_base, mask=mask, other=0.0)
        
        normed = (val - mean) * inv_std
        result = normed * gn_w + gn_b
        
        tl.store(out_ptr + out_base, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.gn_eps = 1e-5

    def forward(self, x):
        # Run conv_transpose using torch (highly optimized cuDNN)
        x = self.conv_transpose(x)
        
        N, C, H, W = x.shape
        H_out = H // 2
        W_out = W // 2
        
        # Compute fused BN scale/bias
        if self.training:
            # fall back to standard path during training
            x = self.batch_norm(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x
        
        bn = self.batch_norm
        bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bn_bias = bn.bias - bn.running_mean * bn_scale
        
        bn_scale = bn_scale.contiguous()
        bn_bias = bn_bias.contiguous()
        
        x = x.contiguous()
        out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
        
        groups = self.num_groups
        channels_per_group = C // groups
        spatial_out = H_out * W_out
        group_size = channels_per_group * spatial_out
        
        # Choose BLOCK as next power of 2 >= group_size, capped
        BLOCK = 1024
        if group_size <= 256:
            BLOCK = 256
        elif group_size <= 512:
            BLOCK = 512
        elif group_size <= 1024:
            BLOCK = 1024
        else:
            BLOCK = 1024
        
        grid = (N * groups,)
        
        fused_bn_tanh_maxpool_gn_kernel[grid](
            x, out,
            bn_scale, bn_bias,
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            N, C, H, W,
            H_out, W_out,
            GROUPS=groups,
            CHANNELS_PER_GROUP=channels_per_group,
            SPATIAL_OUT=spatial_out,
            GROUP_SIZE=group_size,
            BLOCK=BLOCK,
            EPS=self.gn_eps,
            num_warps=4,
        )
        
        return out