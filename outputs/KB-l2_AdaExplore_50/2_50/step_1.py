import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, ic, id, ih, iw_block)
    pid = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    pid_oc = tl.program_id(2)

    n = pid // IC
    ic = pid % IC

    iw = pid_spatial % IW
    tmp = pid_spatial // IW
    ih = tmp % IH
    id_ = tmp // IH

    # load x value
    x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
    x_val = tl.load(x_ptr + x_off)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # for each kernel position, scatter add
    for kd in range(KD):
        od = id_ * STRIDE_D - PAD_D + kd
        if (od >= 0) & (od < OD):
            for kh in range(KH):
                oh = ih * STRIDE_H - PAD_H + kh
                if (oh >= 0) & (oh < OH):
                    for kw in range(KW):
                        ow = iw * STRIDE_W - PAD_W + kw
                        if (ow >= 0) & (ow < OW):
                            # weight shape: (IC, OC, KD, KH, KW)
                            w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                            w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                            contrib = x_val * w_vals
                            out_off = ((n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
                            tl.atomic_add(out_ptr + out_off, contrib, mask=oc_mask)


@triton.jit
def fused_pool_bias_scale_kernel(
    in_ptr, bias_ptr, conv_bias_ptr, out_ptr,
    N, C, OD, OH, OW,
    POD, POH, POW,
    scale_combined,  # scale1 * scale2 / 8
    scale2,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * POD * POH * POW
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    pw = offs % POW
    tmp = offs // POW
    ph = tmp % POH
    tmp = tmp // POH
    pd = tmp % POD
    tmp = tmp // POD
    c = tmp % C
    n = tmp // C

    base = ((n * C + c) * OD) * OH * OW
    d0 = pd * 2
    h0 = ph * 2
    w0 = pw * 2

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for dd in range(2):
        for hh in range(2):
            for ww in range(2):
                off = base + (d0 + dd) * OH * OW + (h0 + hh) * OW + (w0 + ww)
                v = tl.load(in_ptr + off, mask=mask, other=0.0)
                acc += v

    # conv bias
    cb = tl.load(conv_bias_ptr + c, mask=mask, other=0.0)
    # sum_pool_value + 8*conv_bias gives the actual pooled (pre-scale) sum w/ conv bias added at every elem
    # actually conv bias was added to each element of pre-pool tensor, so after avg pool / 8 still adds cb
    # We folded conv bias out; here in_ptr is conv-no-bias output * 1.
    # acc/8 = avg of conv-no-bias output. Then add conv_bias. Then *scale1. Then +bias *scale2.
    # Equivalently: (acc/8 + cb) * scale1 = acc * scale1/8 + cb*scale1
    # Then + bias_extra; then * scale2
    # Combined: ((acc/8 + cb) * scale1 + bias_extra) * scale2
    bias_val = tl.load(bias_ptr + c, mask=mask, other=0.0)
    # scale_combined = scale1/8
    pooled = acc * scale_combined + cb * (scale_combined * 8.0)
    # wait: (acc/8 + cb) * scale1 = acc*(scale1/8) + cb*scale1
    # let's recompute cleanly
    # I'll just compute directly
    val = (acc / 8.0 + cb) * (scale_combined * 8.0)  # this is *scale1
    val = (val + bias_val) * scale2
    tl.store(out_ptr + offs, val, mask=mask)


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

        # output of conv transpose without bias
        out = torch.zeros((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)

        BLOCK_OC = 16
        # ensure OC fits well
        grid = (N * IC, ID * IH * IW, triton.cdiv(OC, BLOCK_OC))
        conv_transpose3d_scatter_kernel[grid](
            x, weight, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC,
        )

        # pooled output
        POD, POH, POW = OD // 2, OH // 2, OW // 2
        pooled = torch.empty((N, OC, POD, POH, POW), device=x.device, dtype=x.dtype)

        total = N * OC * POD * POH * POW
        BLOCK = 256
        grid2 = (triton.cdiv(total, BLOCK),)
        scale_combined = self.scale1.item() / 8.0
        scale2_val = self.scale2.item()

        bias_flat = self.bias.view(-1).contiguous()
        conv_bias = self.conv_transpose.bias.contiguous()

        fused_pool_bias_scale_kernel[grid2](
            out, bias_flat, conv_bias, pooled,
            N, OC, OD, OH, OW,
            POD, POH, POW,
            scale_combined, scale2_val,
            BLOCK=BLOCK,
        )

        return pooled