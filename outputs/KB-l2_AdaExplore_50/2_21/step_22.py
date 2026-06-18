import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_bias_scale_sigmoid_kernel(
    x_ptr,          # input (N, IC, H, W)
    w_ptr,          # weight (OC, IC, KH, KW)
    cb_ptr,         # conv bias (OC,)
    bias_ptr,       # (OC,)
    scale_ptr,      # (OC,)
    out_ptr,        # output (N, OC, OH, OW)
    N, IC, H, W,
    OC, OH, OW,
    BLOCK_HW: tl.constexpr,
    IC_C: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    OC_C: tl.constexpr,
):
    # one program per (n, oc, hw-tile)
    pid = tl.program_id(0)
    n = tl.program_id(1)

    hw_tiles = tl.cdiv(OH * OW, BLOCK_HW)
    oc = pid // hw_tiles
    tile = pid % hw_tiles

    hw_offs = tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offs < (OH * OW)
    oh = hw_offs // OW
    ow = hw_offs % OW

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    # Loop over IC, KH, KW
    for ic in tl.static_range(0, IC_C):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh
                iw = ow + kw
                in_offs = ((n * IC + ic) * H + ih) * W + iw
                x = tl.load(x_ptr + in_offs, mask=mask_hw, other=0.0)
                w_off = ((oc * IC_C + ic) * KH + kh) * KW + kw
                w = tl.load(w_ptr + w_off)
                acc += x * w

    cb = tl.load(cb_ptr + oc)
    b = tl.load(bias_ptr + oc)
    s = tl.load(scale_ptr + oc)
    acc = acc + cb
    acc = (acc + b) * s
    acc = tl.sigmoid(acc)

    out_off = ((n * OC_C + oc) * OH + oh) * OW + ow
    tl.store(out_ptr + out_off, acc, mask=mask_hw)


@triton.jit
def gn_kernel(
    x_ptr,           # (N, C, HW)  -- the post-sigmoid tensor
    gn_weight_ptr,
    gn_bias_ptr,
    out_ptr,
    N, C, HW,
    num_groups,
    group_size,
    eps,
    BLOCK_HW: tl.constexpr,
    CPG: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups
    c_start = g * CPG

    sum_x = 0.0
    sum_x2 = 0.0

    for ci in tl.static_range(0, CPG):
        c = c_start + ci
        for hw_start in range(0, HW, BLOCK_HW):
            offs = hw_start + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            ptr = x_ptr + n * C * HW + c * HW + offs
            x = tl.load(ptr, mask=mask, other=0.0)
            x = tl.where(mask, x, 0.0)
            sum_x += tl.sum(x, axis=0)
            sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / group_size
    var = sum_x2 / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for ci in tl.static_range(0, CPG):
        c = c_start + ci
        gw = tl.load(gn_weight_ptr + c)
        gb = tl.load(gn_bias_ptr + c)
        for hw_start in range(0, HW, BLOCK_HW):
            offs = hw_start + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            ptr = x_ptr + n * C * HW + c * HW + offs
            x = tl.load(ptr, mask=mask, other=0.0)
            y = (x - mean) * rstd * gw + gb
            tl.store(out_ptr + n * C * HW + c * HW + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        conv_w = self.conv.weight.contiguous()
        conv_b = self.conv.bias.contiguous()
        bias_flat = self.bias.view(-1).contiguous()
        scale_flat = self.scale.view(-1).contiguous()

        # Output of conv+bias+scale+sigmoid
        post_sig = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_HW = 256
        hw_tiles = (OH * OW + BLOCK_HW - 1) // BLOCK_HW
        grid = (OC * hw_tiles, N)

        conv_bias_scale_sigmoid_kernel[grid](
            x, conv_w, conv_b, bias_flat, scale_flat, post_sig,
            N, IC, H, W,
            OC, OH, OW,
            BLOCK_HW=BLOCK_HW,
            IC_C=IC,
            KH=KH,
            KW=KW,
            OC_C=OC,
            num_warps=4,
            num_stages=2,
        )

        # Group norm pass
        out = torch.empty_like(post_sig)
        HW = OH * OW
        channels_per_group = OC // self.num_groups
        group_size = channels_per_group * HW
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()

        BLOCK_HW2 = 1024
        grid2 = (N * self.num_groups,)
        gn_kernel[grid2](
            post_sig, gn_w, gn_b, out,
            N, OC, HW,
            self.num_groups,
            group_size,
            self.eps,
            BLOCK_HW=BLOCK_HW2,
            CPG=channels_per_group,
            num_warps=8,
        )
        return out