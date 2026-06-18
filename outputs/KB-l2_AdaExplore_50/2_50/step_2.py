import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 16}, num_warps=2, num_stages=2),
    ],
    key=['OC', 'IC', 'POD', 'POH', 'POW'],
)
@triton.jit
def fused_convt_pool_bias_scale_kernel(
    x_ptr, w_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    POD, POH, POW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_D: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    scale1, scale2,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N * POD * POH * POW, cdiv(OC, BLOCK_OC))
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)

    pw = pid_sp % POW
    tmp = pid_sp // POW
    ph = tmp % POH
    tmp = tmp // POH
    pd = tmp % POD
    n = tmp // POD

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # iterate over the 2x2x2 pooling window
    for dd in tl.static_range(2):
        od = pd * 2 + dd
        for hh in tl.static_range(2):
            oh = ph * 2 + hh
            for ww in tl.static_range(2):
                ow = pw * 2 + ww
                # iterate over kernel
                for kd in tl.static_range(KD):
                    id_num = od + PAD_D - kd
                    id_ = id_num // STRIDE_D
                    valid_d = ((id_num - id_ * STRIDE_D) == 0) & (id_ >= 0) & (id_ < ID)
                    for kh in tl.static_range(KH):
                        ih_num = oh + PAD_H - kh
                        ih = ih_num // STRIDE_H
                        valid_h = ((ih_num - ih * STRIDE_H) == 0) & (ih >= 0) & (ih < IH)
                        for kw in tl.static_range(KW):
                            iw_num = ow + PAD_W - kw
                            iw = iw_num // STRIDE_W
                            valid_w = ((iw_num - iw * STRIDE_W) == 0) & (iw >= 0) & (iw < IW)
                            valid = valid_d & valid_h & valid_w
                            for ic in tl.static_range(0, 1):
                                pass
                            # loop over ic
                            for ic in range(IC):
                                x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
                                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                                # weight: (IC, OC, KD, KH, KW)
                                w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                                w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                acc += x_val * w_vals

    # add conv bias * 8 (since 8 elements pooled), divide by 8, multiply by scale1
    cb = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    # (acc/8 + cb) * scale1 = acc * scale1/8 + cb * scale1
    val = acc * (scale1 / 8.0) + cb * scale1
    val = (val + b) * scale2

    out_off = (((n * OC + oc_offs) * POD + pd) * POH + ph) * POW + pw
    tl.store(out_ptr + out_off, val, mask=oc_mask)


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
        self.kernel_size_t = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride_t = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding_t = padding if isinstance(padding, tuple) else (padding,) * 3

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD, KH, KW = self.kernel_size_t
        SD, SH, SW = self.stride_t
        PD, PH, PW = self.padding_t
        OC = self.out_channels

        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        POD, POH, POW = OD // 2, OH // 2, OW // 2

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        conv_bias = self.conv_transpose.bias.contiguous()
        bias_flat = self.bias.view(-1).contiguous()

        pooled = torch.empty((N, OC, POD, POH, POW), device=x.device, dtype=x.dtype)

        scale1_val = self.scale1.item()
        scale2_val = self.scale2.item()

        grid = lambda META: (N * POD * POH * POW, triton.cdiv(OC, META['BLOCK_OC']))

        fused_convt_pool_bias_scale_kernel[grid](
            x, weight, conv_bias, bias_flat, pooled,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            POD, POH, POW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            scale1_val, scale2_val,
        )

        return pooled