import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_PH': 2, 'BLOCK_PW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_PH': 2, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_PH': 4, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_PH': 1, 'BLOCK_PW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_PH': 1, 'BLOCK_PW': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'POH', 'POW', 'IC'],
)
@triton.jit
def fused_conv_tanh_scale_bias_pool_kernel(
    x_ptr, w_ptr, cb_ptr, bias_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC, OH, OW,
    POH, POW,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_PH: tl.constexpr, BLOCK_PW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_ph = tl.program_id(1)
    pid_pw = tl.program_id(2)

    n = pid_nc // OC
    oc = pid_nc % OC

    ph_start = pid_ph * BLOCK_PH
    pw_start = pid_pw * BLOCK_PW

    # conv output tile dims
    CH: tl.constexpr = BLOCK_PH * POOL
    CW: tl.constexpr = BLOCK_PW * POOL

    oh_start = ph_start * POOL
    ow_start = pw_start * POOL

    offs_ch = tl.arange(0, CH)
    offs_cw = tl.arange(0, CW)

    cb = tl.load(cb_ptr + oc)
    b_extra = tl.load(bias_ptr + oc)

    conv_acc = tl.zeros((CH, CW), dtype=tl.float32)

    # Loop order: ic, kh, kw — load input slice once per (ic, kh, kw)
    for ic in tl.static_range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh_start + offs_ch + kh
                iw = ow_start + offs_cw + kw
                mask_ih = ih < IH
                mask_iw = iw < IW
                in_mask = mask_ih[:, None] & mask_iw[None, :]
                x_off = ((n * IC + ic) * IH + ih[:, None]) * IW + iw[None, :]
                x_val = tl.load(x_ptr + x_off, mask=in_mask, other=0.0)
                w_val = tl.load(w_ptr + ((oc * IC + ic) * KH + kh) * KW + kw)
                conv_acc += x_val * w_val

    # bias + tanh + scale + bias
    conv_acc = conv_acc + cb
    e_pos = tl.exp(conv_acc)
    e_neg = tl.exp(-conv_acc)
    t = (e_pos - e_neg) / (e_pos + e_neg)
    v = t * SCALE + b_extra

    # mask out-of-bounds conv outputs
    oh_full = oh_start + offs_ch
    ow_full = ow_start + offs_cw
    valid = (oh_full[:, None] < OH) & (ow_full[None, :] < OW)
    neg_large = -1.0e30
    v = tl.where(valid, v, neg_large)

    # max-pool: reshape to [BLOCK_PH, POOL, BLOCK_PW, POOL] and reduce over POOL dims
    v4 = tl.reshape(v, (BLOCK_PH, POOL, BLOCK_PW, POOL))
    p1 = tl.max(v4, axis=3)  # [BLOCK_PH, POOL, BLOCK_PW]
    pool_out = tl.max(p1, axis=1)  # [BLOCK_PH, BLOCK_PW]

    offs_ph = ph_start + tl.arange(0, BLOCK_PH)
    offs_pw = pw_start + tl.arange(0, BLOCK_PW)
    out_mask = (offs_ph[:, None] < POH) & (offs_pw[None, :] < POW)
    out_off = ((n * OC + oc) * POH + offs_ph[:, None]) * POW + offs_pw[None, :]
    tl.store(out_ptr + out_off, pool_out, mask=out_mask)


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

        grid = lambda meta: (
            N * OC,
            triton.cdiv(POH, meta['BLOCK_PH']),
            triton.cdiv(POW, meta['BLOCK_PW']),
        )

        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, weight, cb, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.scaling_factor,
        )
        return out