import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['CHANNELS_PER_GROUP', 'POOL_SIZE'],
)
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
    EPS: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    # 2D tile: rows = channels in group, cols = spatial positions
    c_offs = tl.arange(0, CHANNELS_PER_GROUP)  # [CPG]
    s_offs = tl.arange(0, POOL_SIZE)            # [POOL_SIZE]
    
    # decode spatial -> (oh, ow)
    oh = s_offs // OW   # [POOL_SIZE]
    ow = s_offs % OW    # [POOL_SIZE]
    ih0 = 2 * oh
    iw0 = 2 * ow
    
    # channel indices
    c_idx = g * CHANNELS_PER_GROUP + c_offs  # [CPG]
    
    # base offsets per channel: n*C*H*W + c*H*W  -> [CPG, 1]
    nc_base = n * C * H * W + c_idx[:, None] * (H * W)
    # spatial offsets per pos -> [1, POOL_SIZE]
    sp00 = (ih0 * W + iw0)[None, :]
    sp01 = sp00 + 1
    sp10 = sp00 + W
    sp11 = sp10 + 1
    
    p00 = tl.load(x_ptr + nc_base + sp00)
    p01 = tl.load(x_ptr + nc_base + sp01)
    p10 = tl.load(x_ptr + nc_base + sp10)
    p11 = tl.load(x_ptr + nc_base + sp11)
    
    scale = tl.load(bn_scale_ptr + c_idx)[:, None]  # [CPG, 1]
    shift = tl.load(bn_shift_ptr + c_idx)[:, None]
    
    v00 = tl.extra.cuda.libdevice.tanh(p00 * scale + shift)
    v01 = tl.extra.cuda.libdevice.tanh(p01 * scale + shift)
    v10 = tl.extra.cuda.libdevice.tanh(p10 * scale + shift)
    v11 = tl.extra.cuda.libdevice.tanh(p11 * scale + shift)
    
    m1 = tl.maximum(v00, v01)
    m2 = tl.maximum(v10, v11)
    m = tl.maximum(m1, m2)  # [CPG, POOL_SIZE]
    
    # Compute group stats
    sum_val = tl.sum(m)
    sum_sq = tl.sum(m * m)
    inv_n = 1.0 / GROUP_SIZE
    mean = sum_val * inv_n
    var = sum_sq * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)
    
    # Apply GN affine
    gw = tl.load(gn_weight_ptr + c_idx)[:, None]
    gb = tl.load(gn_bias_ptr + c_idx)[:, None]
    
    result = (m - mean) * rstd * gw + gb
    
    # Store
    out_base = n * C * OH * OW + c_idx[:, None] * (OH * OW) + s_offs[None, :]
    tl.store(out_ptr + out_base, result)


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
        self._cached_scale = None
        self._cached_shift = None
        self._cached_gn_w = None
        self._cached_gn_b = None

    def _get_bn_fused(self):
        bn = self.batch_norm
        if self._cached_scale is None or self._cached_scale.device != bn.weight.device:
            with torch.no_grad():
                scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).contiguous()
                shift = (bn.bias - bn.running_mean * scale).contiguous()
            self._cached_scale = scale
            self._cached_shift = shift
            self._cached_gn_w = self.group_norm.weight.contiguous()
            self._cached_gn_b = self.group_norm.bias.contiguous()
        return self._cached_scale, self._cached_shift, self._cached_gn_w, self._cached_gn_b

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
        
        scale, shift, gn_w, gn_b = self._get_bn_fused()
        
        x = x.contiguous()
        N, C, H, W = x.shape
        OH = H // 2
        OW = W // 2
        
        out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
        
        channels_per_group = C // self.num_groups
        pool_size = OH * OW
        group_size = channels_per_group * pool_size
        
        grid = (N * self.num_groups,)
        
        fused_bn_tanh_maxpool_gn_kernel[grid](
            x, out,
            scale, shift,
            gn_w, gn_b,
            N, C, H, W, OH, OW,
            GROUPS=self.num_groups,
            CHANNELS_PER_GROUP=channels_per_group,
            POOL_SIZE=pool_size,
            GROUP_SIZE=group_size,
            EPS=self.group_norm.eps,
        )
        
        return out