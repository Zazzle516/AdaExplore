import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_conv_bn_tanh_pool_kernel(
    x_ptr,           # input: [N, IC, IH, IW]
    w_ptr,           # weight: [IC, OC, KH, KW] (ConvTranspose2d weight layout)
    scale_ptr,       # bn fused scale [OC]
    shift_ptr,       # bn fused shift [OC]
    out_ptr,         # output: [N, OC, OH_pool, OW_pool]
    N, IC, IH, IW,
    OC, OH, OW,
    OH_POOL, OW_POOL,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,  # spatial block in pooled output
):
    # Each program handles one (n, oc_block, sp_block) of pooled output
    pid = tl.program_id(0)
    n = tl.program_id(1)
    oc_block_id = tl.program_id(2)

    sp_block_id = pid
    sp_start = sp_block_id * BLOCK_SP
    sp_offs = sp_start + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < (OH_POOL * OW_POOL)

    # pooled output positions
    oh_p = sp_offs // OW_POOL
    ow_p = sp_offs % OW_POOL

    oc_start = oc_block_id * BLOCK_OC
    oc_offs = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # We need to compute 4 conv outputs per pooled position: (2*oh_p, 2*ow_p), (2*oh_p, 2*ow_p+1), (2*oh_p+1, 2*ow_p), (2*oh_p+1, 2*ow_p+1)
    # And max-pool them after bn+tanh.

    # accumulators for 4 outputs, shape [BLOCK_SP, BLOCK_OC]
    acc00 = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # Output coords for the 4 positions (in conv output, before pooling)
    oh0 = oh_p * 2
    ow0 = ow_p * 2
    oh1 = oh0 + 1
    ow1 = ow0 + 1

    # ConvTranspose2d with stride=1, padding=PAD is equivalent to:
    # out[oh, ow] = sum_{ic, kh, kw} x[ic, oh + PAD - kh, ow + PAD - kw] * w[ic, oc, kh, kw]
    # i.e., a regular convolution with the weight flipped, with effective padding = (KH-1-PAD)
    # Using direct formula: ih = oh + PAD - kh, iw = ow + PAD - kw

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # input positions for each of the 4 outputs
            ih00 = oh0 + PAD - kh
            iw00 = ow0 + PAD - kw
            ih01 = oh0 + PAD - kh
            iw01 = ow1 + PAD - kw
            ih10 = oh1 + PAD - kh
            iw10 = ow0 + PAD - kw
            ih11 = oh1 + PAD - kh
            iw11 = ow1 + PAD - kw

            valid00 = (ih00 >= 0) & (ih00 < IH) & (iw00 >= 0) & (iw00 < IW)
            valid01 = (ih01 >= 0) & (ih01 < IH) & (iw01 >= 0) & (iw01 < IW)
            valid10 = (ih10 >= 0) & (ih10 < IH) & (iw10 >= 0) & (iw10 < IW)
            valid11 = (ih11 >= 0) & (ih11 < IH) & (iw11 >= 0) & (iw11 < IW)

            # Loop over IC in tiles
            BLOCK_IC: tl.constexpr = 64
            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                # Load weights w[ic, oc, kh, kw] -> shape [BLOCK_IC, BLOCK_OC]
                w_ptrs = w_ptr + ic_offs[:, None] * (OC * KH * KW) + oc_offs[None, :] * (KH * KW) + kh * KW + kw
                w_vals = tl.load(w_ptrs, mask=ic_mask[:, None] & oc_mask[None, :], other=0.0)

                # Load x for the 4 positions: shape [BLOCK_SP, BLOCK_IC]
                x_base = n * (IC * IH * IW)
                # position 00
                x_ptrs00 = x_ptr + x_base + ic_offs[None, :] * (IH * IW) + ih00[:, None] * IW + iw00[:, None]
                m00 = sp_mask[:, None] & ic_mask[None, :] & valid00[:, None]
                x00 = tl.load(x_ptrs00, mask=m00, other=0.0)
                acc00 += tl.dot(x00, w_vals)

                x_ptrs01 = x_ptr + x_base + ic_offs[None, :] * (IH * IW) + ih01[:, None] * IW + iw01[:, None]
                m01 = sp_mask[:, None] & ic_mask[None, :] & valid01[:, None]
                x01 = tl.load(x_ptrs01, mask=m01, other=0.0)
                acc01 += tl.dot(x01, w_vals)

                x_ptrs10 = x_ptr + x_base + ic_offs[None, :] * (IH * IW) + ih10[:, None] * IW + iw10[:, None]
                m10 = sp_mask[:, None] & ic_mask[None, :] & valid10[:, None]
                x10 = tl.load(x_ptrs10, mask=m10, other=0.0)
                acc10 += tl.dot(x10, w_vals)

                x_ptrs11 = x_ptr + x_base + ic_offs[None, :] * (IH * IW) + ih11[:, None] * IW + iw11[:, None]
                m11 = sp_mask[:, None] & ic_mask[None, :] & valid11[:, None]
                x11 = tl.load(x_ptrs11, mask=m11, other=0.0)
                acc11 += tl.dot(x11, w_vals)

    # Apply BN fold + tanh + maxpool
    scale = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    shift = tl.load(shift_ptr + oc_offs, mask=oc_mask, other=0.0)

    v00 = acc00 * scale[None, :] + shift[None, :]
    v01 = acc01 * scale[None, :] + shift[None, :]
    v10 = acc10 * scale[None, :] + shift[None, :]
    v11 = acc11 * scale[None, :] + shift[None, :]

    # tanh via 2*sigmoid(2x) - 1 (more stable than exp form)
    # Use: tanh(x) = 1 - 2/(exp(2x)+1)
    e00 = tl.exp(2.0 * v00)
    e01 = tl.exp(2.0 * v01)
    e10 = tl.exp(2.0 * v10)
    e11 = tl.exp(2.0 * v11)
    t00 = 1.0 - 2.0 / (e00 + 1.0)
    t01 = 1.0 - 2.0 / (e01 + 1.0)
    t10 = 1.0 - 2.0 / (e10 + 1.0)
    t11 = 1.0 - 2.0 / (e11 + 1.0)

    m0 = tl.maximum(t00, t01)
    m1 = tl.maximum(t10, t11)
    pooled = tl.maximum(m0, m1)

    # Store to output [N, OC, OH_POOL, OW_POOL]
    out_base = n * (OC * OH_POOL * OW_POOL)
    out_ptrs = out_ptr + out_base + oc_offs[None, :] * (OH_POOL * OW_POOL) + sp_offs[:, None]
    tl.store(out_ptrs, pooled, mask=sp_mask[:, None] & oc_mask[None, :])


@triton.jit
def group_norm_kernel(
    x_ptr,           # [N, C, H, W]
    out_ptr,         # [N, C, H, W]
    gw_ptr,          # [C]
    gb_ptr,          # [C]
    N, C, H, W,
    G,
    CPG: tl.constexpr,
    HW: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    c_base = g * CPG
    total = CPG * HW

    sum_val = 0.0
    sum_sq = 0.0

    base = n * C * HW + c_base * HW

    for ci in tl.static_range(0, CPG):
        for blk_start in range(0, HW, BLOCK):
            offs = blk_start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + base + ci * HW + offs, mask=mask, other=0.0)
            sum_val += tl.sum(v)
            sum_sq += tl.sum(v * v)

    mean = sum_val / total
    var = sum_sq / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for ci in tl.static_range(0, CPG):
        c = c_base + ci
        gw = tl.load(gw_ptr + c)
        gb = tl.load(gb_ptr + c)
        for blk_start in range(0, HW, BLOCK):
            offs = blk_start + tl.arange(0, BLOCK)
            mask = offs < HW
            v = tl.load(x_ptr + base + ci * HW + offs, mask=mask, other=0.0)
            v = (v - mean) * rstd * gw + gb
            tl.store(out_ptr + base + ci * HW + offs, v, mask=mask)


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
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        if self.training or self.stride != 1:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = self.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        PAD = self.padding

        # ConvTranspose2d output spatial size with stride=1
        OH = IH + KH - 1 - 2 * PAD
        OW = IW + KW - 1 - 2 * PAD
        OH_POOL = OH // 2
        OW_POOL = OW // 2

        # BN fold (eval mode)
        bn = self.batch_norm
        scale_bn = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        shift_bn = bn.bias - bn.running_mean * scale_bn

        # Add conv bias to shift
        if self.conv_transpose.bias is not None:
            shift = shift_bn + self.conv_transpose.bias * scale_bn
        else:
            shift = shift_bn
        scale = scale_bn

        # Weight: ConvTranspose2d weight is [IC, OC, KH, KW]
        # We need to flip kh, kw because the equivalent conv uses flipped kernel
        # ConvTranspose with stride=1, padding=p: out[h, w] = sum x[h+p-kh, w+p-kw] * w[ic, oc, kh, kw]
        # That's our formula already, no flip needed in this index math.
        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]

        x = x.contiguous()
        out_pooled = torch.empty((N, OC, OH_POOL, OW_POOL), device=x.device, dtype=x.dtype)

        BLOCK_OC = 128
        BLOCK_SP = 32

        sp_blocks = (OH_POOL * OW_POOL + BLOCK_SP - 1) // BLOCK_SP
        oc_blocks = (OC + BLOCK_OC - 1) // BLOCK_OC

        grid = (sp_blocks, N, oc_blocks)
        fused_conv_bn_tanh_pool_kernel[grid](
            x, weight, scale.contiguous(), shift.contiguous(), out_pooled,
            N, IC, IH, IW,
            OC, OH, OW,
            OH_POOL, OW_POOL,
            KH, KW, PAD,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            num_warps=8,
            num_stages=2,
        )

        # GroupNorm
        G = self.num_groups
        CPG = OC // G
        HW = OH_POOL * OW_POOL

        out = torch.empty_like(out_pooled)
        BLOCK = 256
        if HW <= 64:
            BLOCK = 64
        elif HW <= 256:
            BLOCK = 256
        else:
            BLOCK = 512

        grid_gn = (N * G,)
        group_norm_kernel[grid_gn](
            out_pooled, out,
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            N, OC, OH_POOL, OW_POOL,
            G, CPG, HW,
            self.group_norm.eps,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out