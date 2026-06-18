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
    BLOCK_PW: tl.constexpr,
):
    # BLOCK_PW = number of pool windows per program
    # Inner spatial dim = BLOCK_PW * POOL * POOL conv positions
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_pw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    pw_offs = pid_pw * BLOCK_PW + tl.arange(0, BLOCK_PW)  # [BLOCK_PW] pool-window indices

    oc_mask = oc_offs < OC
    pw_mask = pw_offs < (POH * POW)

    # pool window coords
    poh = pw_offs // POW
    pow_ = pw_offs % POW

    # Conv-output positions within each window: POOL*POOL per pool window
    POOL2: tl.constexpr = POOL * POOL
    pp_offs = tl.arange(0, POOL2)  # [POOL2]
    pp_h = pp_offs // POOL  # 0..POOL-1
    pp_w = pp_offs % POOL

    # Flattened spatial inside this program: [BLOCK_PW, POOL2]
    # conv output (oh, ow) = (poh*POOL + pp_h, pow*POOL + pp_w)
    oh = poh[:, None] * POOL + pp_h[None, :]  # [BLOCK_PW, POOL2]
    ow = pow_[:, None] * POOL + pp_w[None, :]  # [BLOCK_PW, POOL2]

    # accumulator
    acc = tl.zeros((BLOCK_OC, BLOCK_PW, POOL2), dtype=tl.float32)

    # Load biases
    conv_b = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    extra_b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)

    x_n_base = pid_n * (IC * IH * IW)
    w_oc_base = oc_offs * (IC * KH * KW)  # [BLOCK_OC]

    # Loop over (ic, kh, kw): load weight once, then accumulate across all spatial positions
    for ic in range(0, IC):
        x_ic_base = x_n_base + ic * (IH * IW)
        w_ic_base = w_oc_base + ic * (KH * KW)
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                w_off = w_ic_base + kh * KW + kw  # [BLOCK_OC]
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                ih = oh + kh  # [BLOCK_PW, POOL2]
                iw = ow + kw
                x_off = x_ic_base + ih * IW + iw  # [BLOCK_PW, POOL2]
                x_mask = pw_mask[:, None]
                x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_PW, POOL2]
                acc += w_val[:, None, None] * x_val[None, :, :]

    # add conv bias
    acc = acc + conv_b[:, None, None]
    # tanh
    ex = tl.exp(acc)
    enx = tl.exp(-acc)
    t = (ex - enx) / (ex + enx)
    t = t * SCALE + extra_b[:, None, None]

    # max over POOL2 dim
    max_val = tl.max(t, axis=2)  # [BLOCK_OC, BLOCK_PW]

    # store: out[n, oc, pw_offs] flat
    out_off = pid_n * (OC * POH * POW) + oc_offs[:, None] * (POH * POW) + pw_offs[None, :]
    out_mask = oc_mask[:, None] & pw_mask[None, :]
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

        BLOCK_OC = 32
        BLOCK_PW = 8

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(POH * POW, BLOCK_PW))

        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, w, cb, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.scaling_factor,
            BLOCK_OC, BLOCK_PW,
            num_warps=4, num_stages=2,
        )
        return out