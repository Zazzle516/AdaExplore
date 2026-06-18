import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW', 'K'],
)
@triton.jit
def conv_transpose_bn_tanh_kernel(
    x_ptr,        # [N, IC, IH, IW]
    w_ptr,        # flipped weight [OC, IC, K, K]
    bias_ptr,     # fused bias [OC]
    scale_ptr,    # bn scale [OC]
    shift_ptr,    # bn shift [OC]
    out_ptr,      # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    K: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # one program per (n, oc_tile, hw_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    HW = OH * OW
    
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    
    oc_mask = offs_oc < OC
    hw_mask = offs_hw < HW
    
    oh = offs_hw // OW
    ow = offs_hw % OW
    
    x_batch_off = pid_n * IC * IH * IW
    
    # accumulator [BLOCK_OC, BLOCK_HW]
    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)
    
    # iterate over kh, kw outer (compute ih, iw once); IC innermost
    for kh in range(0, K):
        ih = oh + kh - PAD
        ih_ok = (ih >= 0) & (ih < IH)
        for kw in range(0, K):
            iw = ow + kw - PAD
            in_bounds = ih_ok & (iw >= 0) & (iw < IW) & hw_mask
            spatial_off = ih * IW + iw
            for ic in range(0, IC):
                x_off = x_batch_off + ic * IH * IW + spatial_off
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)  # [BLOCK_HW]
                
                w_off = offs_oc * (IC * K * K) + ic * K * K + kh * K + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                
                acc += w_val[:, None] * x_val[None, :]
    
    # apply fused bias + bn scale/shift + tanh
    bias = tl.load(bias_ptr + offs_oc, mask=oc_mask, other=0.0)
    scale = tl.load(scale_ptr + offs_oc, mask=oc_mask, other=0.0)
    shift = tl.load(shift_ptr + offs_oc, mask=oc_mask, other=0.0)
    
    # acc is conv output before bias; full pre-tanh = (acc + bias) * scale_bn + shift_bn
    # but we folded bias into shift already if we want; simpler: pass conv_bias and bn separately
    # final = tanh((acc + bias) * scale + shift)
    val = (acc + bias[:, None]) * scale[:, None] + shift[:, None]
    val = tl.extra.cuda.libdevice.tanh(val)
    
    out_off = pid_n * OC * HW + offs_oc[:, None] * HW + offs_hw[None, :]
    out_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_off, val, mask=out_mask)


@triton.jit
def fused_maxpool_gn_kernel(
    x_ptr,           # input [N, C, H, W] (post tanh)
    out_ptr,         # output [N, C, OH, OW]
    gn_weight_ptr,   # [C]
    gn_bias_ptr,     # [C]
    N, C, H, W, OH, OW,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS
    
    # Single-pass: load entire group into SRAM
    idx = tl.arange(0, BLOCK)
    mask = idx < GROUP_SIZE
    
    c_local = idx // POOL_SIZE
    spatial = idx % POOL_SIZE
    oh = spatial // OW
    ow = spatial % OW
    c = g * CHANNELS_PER_GROUP + c_local
    
    ih0 = 2 * oh
    iw0 = 2 * ow
    base = n * C * H * W + c * H * W
    
    p00 = tl.load(x_ptr + base + ih0 * W + iw0, mask=mask, other=0.0)
    p01 = tl.load(x_ptr + base + ih0 * W + iw0 + 1, mask=mask, other=0.0)
    p10 = tl.load(x_ptr + base + (ih0 + 1) * W + iw0, mask=mask, other=0.0)
    p11 = tl.load(x_ptr + base + (ih0 + 1) * W + iw0 + 1, mask=mask, other=0.0)
    
    m1 = tl.maximum(p00, p01)
    m2 = tl.maximum(p10, p11)
    m = tl.maximum(m1, m2)
    m = tl.where(mask, m, 0.0)
    
    sum_val = tl.sum(m, axis=0)
    sum_sq = tl.sum(m * m, axis=0)
    
    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)
    
    gw = tl.load(gn_weight_ptr + c, mask=mask, other=0.0)
    gb = tl.load(gn_bias_ptr + c, mask=mask, other=0.0)
    
    normed = (m - mean) * rstd
    result = normed * gw + gb
    
    out_offset = n * C * OH * OW + c * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_offset, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups
        # stride must be 1 for our equivalence
        assert stride == 1, "Only stride=1 supported in this optimized kernel"
        
        # New padding for equivalent regular conv
        self.equiv_pad = kernel_size - 1 - padding
        
        self._cached = False

    def _build_cache(self, device, dtype):
        # ConvTranspose2d weight shape: [in_channels, out_channels, K, K]
        # Equivalent conv2d: weight shape [out_channels, in_channels, K, K]
        # with weights flipped along H and W axes.
        w = self.conv_transpose.weight.data  # [IC, OC, K, K]
        # transpose to [OC, IC, K, K]
        w_t = w.permute(1, 0, 2, 3).contiguous()
        # flip last two dims
        w_flipped = torch.flip(w_t, dims=[2, 3]).contiguous()
        
        self.register_buffer('_w_flipped', w_flipped, persistent=False)
        
        bn = self.batch_norm
        bn_var = bn.running_var
        bn_mean = bn.running_mean
        bn_weight = bn.weight
        bn_bias = bn.bias
        bn_eps = bn.eps
        
        scale = bn_weight / torch.sqrt(bn_var + bn_eps)
        shift = bn_bias - bn_mean * scale
        
        conv_bias = self.conv_transpose.bias
        if conv_bias is None:
            conv_bias = torch.zeros(self.out_channels, device=device, dtype=dtype)
        
        self.register_buffer('_conv_bias', conv_bias.contiguous(), persistent=False)
        self.register_buffer('_bn_scale', scale.contiguous(), persistent=False)
        self.register_buffer('_bn_shift', shift.contiguous(), persistent=False)
        self.register_buffer('_gn_weight', self.group_norm.weight.data.contiguous(), persistent=False)
        self.register_buffer('_gn_bias', self.group_norm.bias.data.contiguous(), persistent=False)
        
        self._cached = True

    def forward(self, x):
        if self.training or self.batch_norm.training:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x
        
        if not self._cached:
            self._build_cache(x.device, x.dtype)
        
        x = x.contiguous().to(x.device)
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        K = self.kernel_size
        PAD = self.equiv_pad
        # output H/W of conv transpose with stride=1: OH = IH - 2*padding + (K-1)
        # equivalently regular conv with PAD=K-1-padding gives same: OH = IH + 2*PAD - K + 1 = IH + 2*(K-1-padding) - K + 1 = IH + K - 1 - 2*padding
        OH = IH + 2 * PAD - K + 1
        OW = IW + 2 * PAD - K + 1
        
        conv_out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
        
        grid = lambda META: (
            N,
            triton.cdiv(OC, META['BLOCK_OC']),
            triton.cdiv(OH * OW, META['BLOCK_HW']),
        )
        
        conv_transpose_bn_tanh_kernel[grid](
            x, self._w_flipped, self._conv_bias, self._bn_scale, self._bn_shift,
            conv_out,
            N, IC, IH, IW, OC, OH, OW,
            K=K, PAD=PAD,
        )
        
        # MaxPool + GroupNorm
        POH = OH // 2
        POW = OW // 2
        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)
        
        channels_per_group = OC // self.num_groups
        pool_size = POH * POW
        group_size = channels_per_group * pool_size
        
        # Pick BLOCK as next power of 2 >= group_size so we can do single-pass GN
        BLOCK = 1
        while BLOCK < group_size:
            BLOCK *= 2
        
        grid2 = (N * self.num_groups,)
        
        fused_maxpool_gn_kernel[grid2](
            conv_out, out,
            self._gn_weight, self._gn_bias,
            N, OC, OH, OW, POH, POW,
            GROUPS=self.num_groups,
            CHANNELS_PER_GROUP=channels_per_group,
            POOL_SIZE=pool_size,
            GROUP_SIZE=group_size,
            BLOCK=BLOCK,
            EPS=self.group_norm.eps,
            num_warps=8,
        )
        
        return out