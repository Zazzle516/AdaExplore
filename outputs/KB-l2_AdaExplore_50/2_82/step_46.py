import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_scale_bias_pool_kernel(
    x_ptr, w_ptr, cb_ptr, bias_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC: tl.constexpr, OH, OW,
    POH, POW,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_PH: tl.constexpr, BLOCK_PW: tl.constexpr,
):
    # program ids: (n * (OC/BLOCK_OC), ph_block, pw_block)
    pid_no = tl.program_id(0)
    pid_ph = tl.program_id(1)
    pid_pw = tl.program_id(2)

    OC_BLOCKS = OC // BLOCK_OC
    n = pid_no // OC_BLOCKS
    oc_block = pid_no % OC_BLOCKS
    oc_start = oc_block * BLOCK_OC

    offs_oc = oc_start + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]

    ph_start = pid_ph * BLOCK_PH
    pw_start = pid_pw * BLOCK_PW

    offs_ph = ph_start + tl.arange(0, BLOCK_PH)  # [BLOCK_PH]
    offs_pw = pw_start + tl.arange(0, BLOCK_PW)  # [BLOCK_PW]

    # Load conv bias and bias param scalars for this oc tile
    cb = tl.load(cb_ptr + offs_oc)        # [BLOCK_OC]
    b_extra = tl.load(bias_ptr + offs_oc) # [BLOCK_OC]

    neg_large = -1.0e30

    # We'll compute the conv output for the spatial region
    # corresponding to the pool block: spatial size = (BLOCK_PH*POOL, BLOCK_PW*POOL)
    CONV_H: tl.constexpr = BLOCK_PH * POOL
    CONV_W: tl.constexpr = BLOCK_PW * POOL

    offs_oh = ph_start * POOL + tl.arange(0, CONV_H)  # [CONV_H]
    offs_ow = pw_start * POOL + tl.arange(0, CONV_W)  # [CONV_W]

    mask_oh = offs_oh < OH
    mask_ow = offs_ow < OW

    # accumulator for conv, shape [BLOCK_OC, CONV_H, CONV_W]
    conv_acc = tl.zeros((BLOCK_OC, CONV_H, CONV_W), dtype=tl.float32)

    for ic in tl.static_range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = offs_oh + kh  # [CONV_H]
                iw = offs_ow + kw  # [CONV_W]
                # Load x[n, ic, ih, iw] -> shape [CONV_H, CONV_W]
                x_off = ((n * IC + ic) * IH + ih[:, None]) * IW + iw[None, :]
                in_mask = (ih[:, None] < IH) & (iw[None, :] < IW)
                x_val = tl.load(x_ptr + x_off, mask=in_mask, other=0.0)  # [CONV_H, CONV_W]

                # Load w[oc, ic, kh, kw] -> [BLOCK_OC]
                w_off = ((offs_oc * IC + ic) * KH + kh) * KW + kw
                w_val = tl.load(w_ptr + w_off)  # [BLOCK_OC]

                # outer-product accumulate
                conv_acc += w_val[:, None, None] * x_val[None, :, :]

    # Add conv bias
    conv_acc = conv_acc + cb[:, None, None]

    # tanh via 2*sigmoid(2x)-1
    two_x = 2.0 * conv_acc
    t = 2.0 * tl.sigmoid(two_x) - 1.0
    v = t * SCALE + b_extra[:, None, None]

    # mask invalid spatial positions
    valid = mask_oh[None, :, None] & mask_ow[None, None, :]
    v = tl.where(valid, v, neg_large)

    # max pool with POOL x POOL window
    # reshape v from [BLOCK_OC, CONV_H, CONV_W] to [BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW, POOL]
    v_r = tl.reshape(v, (BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW, POOL))
    # max over POOL axes
    pool_acc = tl.max(tl.max(v_r, axis=4), axis=2)  # [BLOCK_OC, BLOCK_PH, BLOCK_PW]

    # store
    out_mask_h = offs_ph < POH
    out_mask_w = offs_pw < POW
    out_mask = out_mask_h[None, :, None] & out_mask_w[None, None, :]
    out_off = ((n * OC + offs_oc[:, None, None]) * POH + offs_ph[None, :, None]) * POW + offs_pw[None, None, :]
    tl.store(out_ptr + out_off, pool_acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.pool_kernel_size
        POH = OH // POOL
        POW = OW // POOL

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

        weight = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()
        bias = self.bias.contiguous().view(-1)

        BLOCK_OC = 16
        BLOCK_PH = 4
        BLOCK_PW = 8

        assert OC % BLOCK_OC == 0

        grid = (
            N * (OC // BLOCK_OC),
            (POH + BLOCK_PH - 1) // BLOCK_PH,
            (POW + BLOCK_PW - 1) // BLOCK_PW,
        )

        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, weight, cb, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.scaling_factor,
            BLOCK_OC,
            BLOCK_PH, BLOCK_PW,
            num_warps=4,
            num_stages=2,
        )
        return out