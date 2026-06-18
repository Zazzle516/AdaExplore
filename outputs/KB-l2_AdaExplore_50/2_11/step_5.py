import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_bn_tanh_pool_kernel(
    x_ptr,        # [N, IC, IH, IW]
    w_ptr,        # [OC, IC, K, K] flipped
    bias_ptr,     # [OC]
    scale_ptr,    # [OC]
    shift_ptr,    # [OC]
    out_ptr,      # [N, OC, POH, POW] - post-pool
    N, IC, IH, IW,
    OC, OH, OW, POH, POW,
    K: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_PH: tl.constexpr,  # pooled height tile
    BLOCK_PW: tl.constexpr,  # pooled width tile
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)
    
    n_pw_tiles = (POW + BLOCK_PW - 1) // BLOCK_PW
    pid_ph = pid_p // n_pw_tiles
    pid_pw = pid_p % n_pw_tiles
    
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_ph = pid_ph * BLOCK_PH + tl.arange(0, BLOCK_PH)
    offs_pw = pid_pw * BLOCK_PW + tl.arange(0, BLOCK_PW)
    
    oc_mask = offs_oc < OC
    ph_mask = offs_ph < POH
    pw_mask = offs_pw < POW
    
    # for each pooled location, we compute 4 conv outputs (2x2 window)
    # output spatial coords: (2*ph + dh, 2*pw + dw), dh,dw in {0,1}
    # We accumulate 4 separate accumulators
    
    # shape [BLOCK_OC, BLOCK_PH, BLOCK_PW]
    acc00 = tl.zeros((BLOCK_OC, BLOCK_PH, BLOCK_PW), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_OC, BLOCK_PH, BLOCK_PW), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_OC, BLOCK_PH, BLOCK_PW), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_OC, BLOCK_PH, BLOCK_PW), dtype=tl.float32)
    
    # output coords for each of 4 sub-positions
    oh0 = 2 * offs_ph         # [BLOCK_PH]
    oh1 = 2 * offs_ph + 1
    ow0 = 2 * offs_pw         # [BLOCK_PW]
    ow1 = 2 * offs_pw + 1
    
    for ic in range(0, IC):
        for kh in range(0, K):
            for kw in range(0, K):
                # weight [BLOCK_OC]
                w_off = offs_oc * (IC * K * K) + ic * K * K + kh * K + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                
                # for each sub-position (dh, dw)
                # ih = oh + kh - PAD, iw = ow + kw - PAD
                base = pid_n * IC * IH * IW + ic * IH * IW
                
                # sub-position (0, 0): oh0, ow0
                ih00 = oh0 + kh - PAD  # [BLOCK_PH]
                iw00 = ow0 + kw - PAD  # [BLOCK_PW]
                ih00_ok = (ih00 >= 0) & (ih00 < IH)
                iw00_ok = (iw00 >= 0) & (iw00 < IW)
                x_off00 = ih00[:, None] * IW + iw00[None, :]
                m00 = ih00_ok[:, None] & iw00_ok[None, :]
                x_val00 = tl.load(x_ptr + base + x_off00, mask=m00, other=0.0)
                acc00 += w_val[:, None, None] * x_val00[None, :, :]
                
                # sub-position (0, 1): oh0, ow1
                ih01 = oh0 + kh - PAD
                iw01 = ow1 + kw - PAD
                ih01_ok = (ih01 >= 0) & (ih01 < IH)
                iw01_ok = (iw01 >= 0) & (iw01 < IW)
                x_off01 = ih01[:, None] * IW + iw01[None, :]
                m01 = ih01_ok[:, None] & iw01_ok[None, :]
                x_val01 = tl.load(x_ptr + base + x_off01, mask=m01, other=0.0)
                acc01 += w_val[:, None, None] * x_val01[None, :, :]
                
                # sub-position (1, 0): oh1, ow0
                ih10 = oh1 + kh - PAD
                iw10 = ow0 + kw - PAD
                ih10_ok = (ih10 >= 0) & (ih10 < IH)
                iw10_ok = (iw10 >= 0) & (iw10 < IW)
                x_off10 = ih10[:, None] * IW + iw10[None, :]
                m10 = ih10_ok[:, None] & iw10_ok[None, :]
                x_val10 = tl.load(x_ptr + base + x_off10, mask=m10, other=0.0)
                acc10 += w_val[:, None, None] * x_val10[None, :, :]
                
                # sub-position (1, 1): oh1, ow1
                ih11 = oh1 + kh - PAD
                iw11 = ow1 + kw - PAD
                ih11_ok = (ih11 >= 0) & (ih11 < IH)
                iw11_ok = (iw11 >= 0) & (iw11 < IW)
                x_off11 = ih11[:, None] * IW + iw11[None, :]
                m11 = ih11_ok[:, None] & iw11_ok[None, :]
                x_val11 = tl.load(x_ptr + base + x_off11, mask=m11, other=0.0)
                acc11 += w_val[:, None, None] * x_val11[None, :, :]
    
    bias = tl.load(bias_ptr + offs_oc, mask=oc_mask, other=0.0)
    scale = tl.load(scale_ptr + offs_oc, mask=oc_mask, other=0.0)
    shift = tl.load(shift_ptr + offs_oc, mask=oc_mask, other=0.0)
    
    b = bias[:, None, None]
    s = scale[:, None, None]
    sh = shift[:, None, None]
    
    v00 = tl.extra.cuda.libdevice.tanh((acc00 + b) * s + sh)
    v01 = tl.extra.cuda.libdevice.tanh((acc01 + b) * s + sh)
    v10 = tl.extra.cuda.libdevice.tanh((acc10 + b) * s + sh)
    v11 = tl.extra.cuda.libdevice.tanh((acc11 + b) * s + sh)
    
    # also need to mask out-of-bounds output positions
    # output OH/OW; the 2x sub-positions must be < OH/OW (always true if POH = OH//2)
    
    m1 = tl.maximum(v00, v01)
    m2 = tl.maximum(v10, v11)
    pooled = tl.maximum(m1, m2)
    
    # store to [N, OC, POH, POW]
    out_off = (pid_n * OC * POH * POW
               + offs_oc[:, None, None] * POH * POW
               + offs_ph[None, :, None] * POW
               + offs_pw[None, None, :])
    out_mask = oc_mask[:, None, None] & ph_mask[None, :, None] & pw_mask[None, None, :]
    tl.store(out_ptr + out_off, pooled, mask=out_mask)


@triton.jit
def groupnorm_kernel(
    x_ptr,           # [N, C, H, W]
    out_ptr,
    gn_weight_ptr,
    gn_bias_ptr,
    N, C, H, W,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    SPATIAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS
    
    offs = tl.arange(0, BLOCK)
    
    sum_val = 0.0
    sum_sq = 0.0
    
    base = n * C * SPATIAL + g * CHANNELS_PER_GROUP * SPATIAL
    
    for start in range(0, GROUP_SIZE, BLOCK):
        idx = start + offs
        mask = idx < GROUP_SIZE
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_val += tl.sum(v, axis=0)
        sum_sq += tl.sum(v * v, axis=0)
    
    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)
    
    for start in range(0, GROUP_SIZE, BLOCK):
        idx = start + offs
        mask = idx < GROUP_SIZE
        c_local = idx // SPATIAL
        c = g * CHANNELS_PER_GROUP + c_local
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        gw = tl.load(gn_weight_ptr + c, mask=mask, other=0.0)
        gb = tl.load(gn_bias_ptr + c, mask=mask, other=0.0)
        normed = (v - mean) * rstd
        result = normed * gw + gb
        tl.store(out_ptr + base + idx, result, mask=mask)


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
        assert stride == 1, "Only stride=1 supported"
        
        self.equiv_pad = kernel_size - 1 - padding
        self._cached = False

    def _build_cache(self, device, dtype):
        w = self.conv_transpose.weight.data  # [IC, OC, K, K]
        w_t = w.permute(1, 0, 2, 3).contiguous()
        w_flipped = torch.flip(w_t, dims=[2, 3]).contiguous()
        
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
        
        self.register_buffer('_w_flipped', w_flipped, persistent=False)
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
        
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        K = self.kernel_size
        PAD = self.equiv_pad
        OH = IH + 2 * PAD - K + 1
        OW = IW + 2 * PAD - K + 1
        POH = OH // 2
        POW = OW // 2
        
        pooled = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)
        
        BLOCK_OC = 32
        BLOCK_PH = 4
        BLOCK_PW = 16
        
        n_pw_tiles = (POW + BLOCK_PW - 1) // BLOCK_PW
        n_ph_tiles = (POH + BLOCK_PH - 1) // BLOCK_PH
        
        grid = (
            N,
            triton.cdiv(OC, BLOCK_OC),
            n_ph_tiles * n_pw_tiles,
        )
        
        conv_bn_tanh_pool_kernel[grid](
            x, self._w_flipped, self._conv_bias, self._bn_scale, self._bn_shift,
            pooled,
            N, IC, IH, IW, OC, OH, OW, POH, POW,
            K=K, PAD=PAD,
            BLOCK_OC=BLOCK_OC, BLOCK_PH=BLOCK_PH, BLOCK_PW=BLOCK_PW,
            num_warps=4, num_stages=2,
        )
        
        out = torch.empty_like(pooled)
        channels_per_group = OC // self.num_groups
        spatial = POH * POW
        group_size = channels_per_group * spatial
        
        BLOCK = 1
        while BLOCK < group_size and BLOCK < 1024:
            BLOCK *= 2
        BLOCK = min(BLOCK, 1024)
        
        grid2 = (N * self.num_groups,)
        groupnorm_kernel[grid2](
            pooled, out,
            self._gn_weight, self._gn_bias,
            N, OC, POH, POW,
            GROUPS=self.num_groups,
            CHANNELS_PER_GROUP=channels_per_group,
            SPATIAL=spatial,
            GROUP_SIZE=group_size,
            BLOCK=BLOCK,
            EPS=self.group_norm.eps,
            num_warps=4,
        )
        
        return out