import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 16}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK': 32}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 256}, num_warps=8, num_stages=3),
    ],
    key=['N', 'OC', 'POD', 'POH', 'POW', 'IC', 'KD'],
)
@triton.jit
def fused_convtr_pool_kernel(
    x_ptr, w_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, POD, POH, POW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    scale1, scale2,
    BLOCK: tl.constexpr,
):
    pid_noc = tl.program_id(0)
    pid_pd = tl.program_id(1)
    pid_spatial = tl.program_id(2)

    n = pid_noc // OC
    oc = pid_noc % OC
    pd = pid_pd

    offs = pid_spatial * BLOCK + tl.arange(0, BLOCK)
    total = POH * POW
    mask = offs < total

    pw = offs % POW
    ph = offs // POW

    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # For each (kd, dd): stride divisibility (dd + PD - kd) % SD must be 0
    # since pd*SD is divisible by SD. Same for kh,hh and kw,ww.
    # Hoist w_val load out of dd/hh/ww loops.
    PD_STEP: tl.constexpr = 2 // SD
    PH_STEP: tl.constexpr = 2 // SH
    PW_STEP: tl.constexpr = 2 // SW
    for ic in range(IC):
        for kd in tl.static_range(KD):
            for kh in tl.static_range(KH):
                for kw in tl.static_range(KW):
                    w_off = ((ic * OC + oc) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)
                    for dd in tl.static_range(2):
                        if ((dd + PD - kd) % SD) == 0:
                            id_off = (dd + PD - kd) // SD
                            id_ = pd * PD_STEP + id_off
                            d_ok = (id_ >= 0) & (id_ < ID)
                            for hh in tl.static_range(2):
                                if ((hh + PH - kh) % SH) == 0:
                                    ih_off = (hh + PH - kh) // SH
                                    ih_ = ph * PH_STEP + ih_off
                                    h_ok = (ih_ >= 0) & (ih_ < IH)
                                    for ww in tl.static_range(2):
                                        if ((ww + PW - kw) % SW) == 0:
                                            iw_off = (ww + PW - kw) // SW
                                            iw_ = pw * PW_STEP + iw_off
                                            w_ok = (iw_ >= 0) & (iw_ < IW)
                                            valid = d_ok & h_ok & w_ok & mask
                                            x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih_ * IW + iw_
                                            v = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                                            acc += v * w_val

    cb = tl.load(conv_bias_ptr + oc)
    b = tl.load(bias_ptr + oc)
    # pooled = acc/8 + cb (conv bias added to each of 8 elements, then averaged)
    pooled = acc * 0.125 + cb
    val = (pooled * scale1 + b) * scale2
    out_off = ((n * OC + oc) * POD + pd) * POH * POW + ph * POW + pw
    tl.store(out_ptr + out_off, val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD, KH, KW = self.kernel_size
        SD, SH, SW = self.stride
        PD, PH, PW = self.padding
        OC = self.out_channels

        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        POD, POH, POW = OD // 2, OH // 2, OW // 2
        pooled = torch.empty((N, OC, POD, POH, POW), device=x.device, dtype=x.dtype)

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        bias_flat = self.bias.view(-1).contiguous()
        conv_bias = self.conv_transpose.bias.contiguous()

        scale1_val = self.scale1.item()
        scale2_val = self.scale2.item()

        total_spatial = POH * POW
        grid = lambda META: (N * OC, POD, triton.cdiv(total_spatial, META['BLOCK']))
        fused_convtr_pool_kernel[grid](
            x, weight, conv_bias, bias_flat, pooled,
            N, IC, ID, IH, IW,
            OC, POD, POH, POW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            scale1_val, scale2_val,
        )

        return pooled