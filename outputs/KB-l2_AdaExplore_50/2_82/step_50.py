import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 2, 'BLOCK_W': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_H': 2, 'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_H': 1, 'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 1, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'POH', 'POW', 'IC'],
)
@triton.jit
def fused_conv_tanh_scale_bias_pool_kernel(
    x_ptr, w_ptr, cb_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    POH, POW,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_ph = tl.program_id(1)
    pid_pw = tl.program_id(2)

    n = pid_nc // OC
    oc = pid_nc % OC

    ph_start = pid_ph * BLOCK_H
    pw_start = pid_pw * BLOCK_W

    CONV_H: tl.constexpr = BLOCK_H * POOL
    CONV_W: tl.constexpr = BLOCK_W * POOL
    # Input tile needs CONV + (K-1) elements
    IN_H: tl.constexpr = CONV_H + KH - 1
    IN_W: tl.constexpr = CONV_W + KW - 1

    offs_ch = ph_start * POOL + tl.arange(0, CONV_H)
    offs_cw = pw_start * POOL + tl.arange(0, CONV_W)

    mask_ch = offs_ch < OH
    mask_cw = offs_cw < OW
    out_mask_conv = mask_ch[:, None] & mask_cw[None, :]

    cb = tl.load(cb_ptr + oc)
    b_extra = tl.load(bias_ptr + oc)

    conv_acc = tl.zeros((CONV_H, CONV_W), dtype=tl.float32)

    # Input tile base offsets
    in_off_h = ph_start * POOL + tl.arange(0, IN_H)  # [IN_H]
    in_off_w = pw_start * POOL + tl.arange(0, IN_W)  # [IN_W]
    in_mask_h = in_off_h < IH
    in_mask_w = in_off_w < IW
    in_mask_2d = in_mask_h[:, None] & in_mask_w[None, :]

    for ic in range(0, IC):
        # Load input tile [IN_H, IN_W] once per (ic)
        x_off = ((n * IC + ic) * IH + in_off_h[:, None]) * IW + in_off_w[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=in_mask_2d, other=0.0)

        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                # slice x_tile [kh:kh+CONV_H, kw:kw+CONV_W]
                # Implement via index gather using arange
                xh = tl.arange(0, CONV_H) + kh
                xw = tl.arange(0, CONV_W) + kw
                # gather: reshape x_tile as 1d index
                gather_idx = xh[:, None] * IN_W + xw[None, :]
                x_flat = tl.reshape(x_tile, (IN_H * IN_W,))
                x_val = tl.load(x_ptr + 0 + gather_idx * 0, mask=False, other=0.0)  # placeholder unused
                # Use direct memory load instead (simpler/correct):
                ih = offs_ch[:, None] + kh
                iw = offs_cw[None, :] + kw
                x_off2 = ((n * IC + ic) * IH + ih) * IW + iw
                in_mask2 = (ih < IH) & (iw < IW)
                x_val = tl.load(x_ptr + x_off2, mask=in_mask2, other=0.0)
                w_val = tl.load(w_ptr + ((oc * IC + ic) * KH + kh) * KW + kw)
                conv_acc += x_val * w_val

    conv_acc = conv_acc + cb
    t = 2.0 * tl.sigmoid(2.0 * conv_acc) - 1.0
    v = t * SCALE + b_extra

    neg_large = -1.0e30
    v = tl.where(out_mask_conv, v, neg_large)

    v = tl.reshape(v, (BLOCK_H, POOL, BLOCK_W, POOL))
    v = tl.max(v, axis=3)
    v = tl.max(v, axis=1)

    offs_ph = ph_start + tl.arange(0, BLOCK_H)
    offs_pw = pw_start + tl.arange(0, BLOCK_W)
    out_mask_h = offs_ph < POH
    out_mask_w = offs_pw < POW
    out_mask = out_mask_h[:, None] & out_mask_w[None, :]
    out_off = ((n * OC + oc) * POH + offs_ph[:, None]) * POW + offs_pw[None, :]
    tl.store(out_ptr + out_off, v, mask=out_mask)


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

        grid = lambda META: (
            N * OC,
            (POH + META['BLOCK_H'] - 1) // META['BLOCK_H'],
            (POW + META['BLOCK_W'] - 1) // META['BLOCK_W'],
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