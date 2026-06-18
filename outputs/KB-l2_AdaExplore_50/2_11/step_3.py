import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_bn_tanh_maxpool_gn_kernel(
    x_ptr,           # input: [N, C, H, W] after conv_transpose
    out_ptr,         # output: [N, C, H/2, W/2]
    bn_scale_ptr,    # [C] = bn_weight / sqrt(bn_var + eps)
    bn_shift_ptr,    # [C] = bn_bias - bn_mean * bn_scale
    gn_weight_ptr,   # [C]
    gn_bias_ptr,     # [C]
    N, C, H, W,
    OH, OW,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    POOL_SIZE: tl.constexpr,  # OH * OW
    GROUP_SIZE: tl.constexpr,  # CHANNELS_PER_GROUP * POOL_SIZE
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    # We'll compute the pooled+tanh+bn output for this group, store it,
    # then compute group statistics, then re-load and apply GN.
    
    offs = tl.arange(0, BLOCK)
    
    # First pass: compute pooled output and accumulate sum, sum_sq
    sum_val = 0.0
    sum_sq = 0.0
    
    # Iterate over the group's elements in blocks
    # group elements are organized as [CHANNELS_PER_GROUP, OH, OW]
    for start in range(0, GROUP_SIZE, BLOCK):
        idx = start + offs
        mask = idx < GROUP_SIZE
        
        # decode idx -> (c_local, oh, ow)
        c_local = idx // POOL_SIZE
        spatial = idx % POOL_SIZE
        oh = spatial // OW
        ow = spatial % OW
        
        c = g * CHANNELS_PER_GROUP + c_local
        
        # pool window: input [n, c, 2*oh:2*oh+2, 2*ow:2*ow+2]
        ih0 = 2 * oh
        iw0 = 2 * ow
        
        base = n * C * H * W + c * H * W
        
        p00 = tl.load(x_ptr + base + ih0 * W + iw0, mask=mask, other=0.0)
        p01 = tl.load(x_ptr + base + ih0 * W + iw0 + 1, mask=mask, other=0.0)
        p10 = tl.load(x_ptr + base + (ih0 + 1) * W + iw0, mask=mask, other=0.0)
        p11 = tl.load(x_ptr + base + (ih0 + 1) * W + iw0 + 1, mask=mask, other=0.0)
        
        # apply bn + tanh, then max
        scale = tl.load(bn_scale_ptr + c, mask=mask, other=0.0)
        shift = tl.load(bn_shift_ptr + c, mask=mask, other=0.0)
        
        # tanh is monotonic, so max(tanh(bn(x))) == tanh(bn(max(x)))? 
        # No! Only if scale > 0. Let's not rely on this. Apply per element.
        v00 = tl.extra.cuda.libdevice.tanh(p00 * scale + shift)
        v01 = tl.extra.cuda.libdevice.tanh(p01 * scale + shift)
        v10 = tl.extra.cuda.libdevice.tanh(p10 * scale + shift)
        v11 = tl.extra.cuda.libdevice.tanh(p11 * scale + shift)
        
        m1 = tl.maximum(v00, v01)
        m2 = tl.maximum(v10, v11)
        m = tl.maximum(m1, m2)
        
        m = tl.where(mask, m, 0.0)
        
        # store to output (we'll re-read for normalization)
        out_offset = n * C * OH * OW + c * OH * OW + oh * OW + ow
        tl.store(out_ptr + out_offset, m, mask=mask)
        
        sum_val += tl.sum(m, axis=0)
        sum_sq += tl.sum(m * m, axis=0)
    
    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)
    
    # Second pass: normalize and apply affine
    for start in range(0, GROUP_SIZE, BLOCK):
        idx = start + offs
        mask = idx < GROUP_SIZE
        
        c_local = idx // POOL_SIZE
        spatial = idx % POOL_SIZE
        oh = spatial // OW
        ow = spatial % OW
        c = g * CHANNELS_PER_GROUP + c_local
        
        out_offset = n * C * OH * OW + c * OH * OW + oh * OW + ow
        v = tl.load(out_ptr + out_offset, mask=mask, other=0.0)
        
        gw = tl.load(gn_weight_ptr + c, mask=mask, other=0.0)
        gb = tl.load(gn_bias_ptr + c, mask=mask, other=0.0)
        
        normed = (v - mean) * rstd
        result = normed * gw + gb
        
        tl.store(out_ptr + out_offset, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        self.num_groups = num_groups
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Compute fused BN scale/shift
        bn = self.batch_norm
        if bn.training:
            # fall back to original path during training
            x = self.batch_norm(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x
        
        bn_var = bn.running_var
        bn_mean = bn.running_mean
        bn_weight = bn.weight
        bn_bias = bn.bias
        bn_eps = bn.eps
        
        scale = bn_weight / torch.sqrt(bn_var + bn_eps)
        shift = bn_bias - bn_mean * scale
        
        x = x.contiguous()
        N, C, H, W = x.shape
        OH = H // 2
        OW = W // 2
        
        out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
        
        channels_per_group = C // self.num_groups
        pool_size = OH * OW
        group_size = channels_per_group * pool_size
        
        # Choose BLOCK as next power of 2 up to a cap
        BLOCK = 1
        while BLOCK < group_size and BLOCK < 1024:
            BLOCK *= 2
        BLOCK = min(BLOCK, 1024)
        
        grid = (N * self.num_groups,)
        
        fused_bn_tanh_maxpool_gn_kernel[grid](
            x, out,
            scale.contiguous(), shift.contiguous(),
            self.group_norm.weight.contiguous(), self.group_norm.bias.contiguous(),
            N, C, H, W, OH, OW,
            GROUPS=self.num_groups,
            CHANNELS_PER_GROUP=channels_per_group,
            POOL_SIZE=pool_size,
            GROUP_SIZE=group_size,
            BLOCK=BLOCK,
            EPS=self.group_norm.eps,
            num_warps=4,
        )
        
        return out