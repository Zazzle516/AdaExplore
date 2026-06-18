import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=['GROUP_SIZE'],
)
@triton.jit
def fused_bn_tanh_maxpool_gn_kernel(
    x_ptr,           # input: [N, C, H, W] after conv_transpose
    out_ptr,         # output: [N, C, H/2, W/2]
    bn_scale_ptr,    # [C]
    bn_shift_ptr,    # [C]
    gn_weight_ptr,   # [C]
    gn_bias_ptr,     # [C]
    N, C, H, W,
    OH, OW,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    offs = tl.arange(0, GROUP_SIZE)

    # Layout: lanes walk [c_local, oh, ow] with ow fastest.
    # contiguous lanes within same c_local => spatial reads are stride-2 (coalesced enough).
    c_local = offs // POOL_SIZE
    spatial = offs % POOL_SIZE
    oh = spatial // OW
    ow = spatial % OW
    c = g * CHANNELS_PER_GROUP + c_local

    ih0 = 2 * oh
    iw0 = 2 * ow

    base = n * C * H * W + c * H * W
    a00 = base + ih0 * W + iw0
    p00 = tl.load(x_ptr + a00)
    p01 = tl.load(x_ptr + a00 + 1)
    p10 = tl.load(x_ptr + a00 + W)
    p11 = tl.load(x_ptr + a00 + W + 1)

    scale = tl.load(bn_scale_ptr + c)
    shift = tl.load(bn_shift_ptr + c)

    v00 = tl.extra.cuda.libdevice.tanh(p00 * scale + shift)
    v01 = tl.extra.cuda.libdevice.tanh(p01 * scale + shift)
    v10 = tl.extra.cuda.libdevice.tanh(p10 * scale + shift)
    v11 = tl.extra.cuda.libdevice.tanh(p11 * scale + shift)

    m1 = tl.maximum(v00, v01)
    m2 = tl.maximum(v10, v11)
    m = tl.maximum(m1, m2)

    sum_val = tl.sum(m, axis=0)
    sum_sq = tl.sum(m * m, axis=0)

    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)

    gw = tl.load(gn_weight_ptr + c)
    gb = tl.load(gn_bias_ptr + c)

    result = (m - mean) * rstd * gw + gb

    out_offset = n * C * OH * OW + c * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_offset, result)


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
            EPS=self.group_norm.eps,
        )
        
        return out