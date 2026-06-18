import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_scale_bias_pool_kernel(
    x_ptr, w_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    POH, POW,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (POH * POW)

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    # The pool window is [poh*POOL, poh*POOL+POOL) x [pow*POOL, pow*POOL+POOL)
    # in conv-output coordinates.
    NEG_INF = float(-1e30)
    max_val = tl.full((BLOCK_OC, BLOCK_SP), NEG_INF, dtype=tl.float32)

    # Load conv bias and extra bias for these OCs
    conv_b = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    extra_b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)      # [BLOCK_OC]

    # For each pool position
    for ph_i in tl.static_range(0, POOL):
        for pw_i in tl.static_range(0, POOL):
            oh = poh * POOL + ph_i  # [BLOCK_SP]
            ow = pow_ * POOL + pw_i # [BLOCK_SP]
            # convolution at (oh, ow) for each oc in oc_offs
            acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
            for ic in range(0, IC):
                for kh in tl.static_range(0, KH):
                    for kw in tl.static_range(0, KW):
                        ih = oh + kh  # [BLOCK_SP]
                        iw = ow + kw  # [BLOCK_SP]
                        # x[n, ic, ih, iw]
                        x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                        x_mask = sp_mask  # ih, iw always in range when sp valid
                        x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_SP]
                        # w[oc, ic, kh, kw]
                        w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                        acc += w_val[:, None] * x_val[None, :]
            # add conv bias
            acc = acc + conv_b[:, None]
            # tanh
            t = (tl.exp(acc) - tl.exp(-acc)) / (tl.exp(acc) + tl.exp(-acc))
            t = t * SCALE + extra_b[:, None]
            max_val = tl.maximum(max_val, t)

    # store max_val to out[n, oc, poh, pow]
    out_off = pid_n * (OC * POH * POW) + oc_offs[:, None] * (POH * POW) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, max_val, mask=out_mask)


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
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        OC = self.out_channels
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.pool_kernel_size
        POH = OH // POOL
        POW = OW // POOL

        w = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()
        b = self.bias.view(-1).contiguous()

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 16
        BLOCK_SP = 64

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(POH * POW, BLOCK_SP))

        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, w, cb, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.scaling_factor,
            BLOCK_OC, BLOCK_SP,
            num_warps=4, num_stages=2,
        )
        return out