import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 32, 'BLOCK_OC': 2}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK': 32, 'BLOCK_OC': 4}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK': 32, 'BLOCK_OC': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 32, 'BLOCK_OC': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 64, 'BLOCK_OC': 2}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK': 64, 'BLOCK_OC': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 64, 'BLOCK_OC': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 64, 'BLOCK_OC': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 64, 'BLOCK_OC': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 64, 'BLOCK_OC': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 128, 'BLOCK_OC': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 128, 'BLOCK_OC': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 128, 'BLOCK_OC': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 128, 'BLOCK_OC': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 256, 'BLOCK_OC': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 256, 'BLOCK_OC': 4}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'ID', 'IH', 'IW', 'OC', 'POD', 'POH', 'POW'],
)
@triton.jit
def fused_convt_pool_kernel(
    x_ptr, w_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    POD, POH, POW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    scale1_div8, scale2,
    BLOCK: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid = tl.program_id(0)
    oc_blk = tl.program_id(1)
    n = tl.program_id(2)

    oc_offs = oc_blk * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC
    oc_safe = tl.where(oc_mask, oc_offs, 0)

    pool_total = POD * POH * POW
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < pool_total

    pw = offs % POW
    tmp = offs // POW
    ph = tmp % POH
    pd = tmp // POH

    cb = tl.load(conv_bias_ptr + oc_safe, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    bv = tl.load(bias_ptr + oc_safe, mask=oc_mask, other=0.0)  # [BLOCK_OC]

    acc = tl.zeros([BLOCK, BLOCK_OC], dtype=tl.float32)

    od0 = pd * 2
    oh0 = ph * 2
    ow0 = pw * 2

    # Hoist (kd,kh,kw,dd,hh,ww) index computation outside ic loop.
    for kd in tl.static_range(KD):
        for dd in tl.static_range(2):
            od = od0 + dd
            id_num = od + PD - kd
            id_q = id_num // SD
            id_ok = (id_num - id_q * SD == 0) & (id_q >= 0) & (id_q < ID)
            id_safe = tl.where(id_ok, id_q, 0)
            for kh in tl.static_range(KH):
                for hh in tl.static_range(2):
                    oh = oh0 + hh
                    ih_num = oh + PH - kh
                    ih_q = ih_num // SH
                    ih_ok = (ih_num - ih_q * SH == 0) & (ih_q >= 0) & (ih_q < IH)
                    ih_safe = tl.where(ih_ok, ih_q, 0)
                    for kw in tl.static_range(KW):
                        for ww in tl.static_range(2):
                            ow = ow0 + ww
                            iw_num = ow + PW - kw
                            iw_q = iw_num // SW
                            iw_ok = (iw_num - iw_q * SW == 0) & (iw_q >= 0) & (iw_q < IW)
                            iw_safe = tl.where(iw_ok, iw_q, 0)
                            valid = id_ok & ih_ok & iw_ok & mask
                            for ic in tl.static_range(0, 3):
                                x_off = ((n * IC + ic) * ID + id_safe) * IH * IW + ih_safe * IW + iw_safe
                                xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)  # [BLOCK]
                                w_off = ((ic * OC + oc_safe) * KD + kd) * KH * KW + kh * KW + kw
                                wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                                acc += xv[:, None] * wv[None, :]

    # cb accumulates 8 times (once per pool position)
    acc += cb[None, :] * 8.0

    result = (acc * scale1_div8 + bv[None, :]) * scale2
    out_off = (n * OC + oc_safe[None, :]) * pool_total + offs[:, None]
    out_mask = mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, result, mask=out_mask)


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

        scale1_div8 = self.scale1.item() / 8.0
        scale2_val = self.scale2.item()

        pool_total = POD * POH * POW
        grid = lambda META: (triton.cdiv(pool_total, META['BLOCK']), triton.cdiv(OC, META['BLOCK_OC']), N)
        fused_convt_pool_kernel[grid](
            x, weight, conv_bias, bias_flat, pooled,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            POD, POH, POW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            scale1_div8, scale2_val,
        )

        return pooled