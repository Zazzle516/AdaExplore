import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, cb_ptr, scale_ptr, bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    OC_C: tl.constexpr,
    IC_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    OHW = OH * OW
    ODHW = OD * OHW

    sp_off = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_off < ODHW

    od = sp_off // OHW
    rem = sp_off % OHW
    oh = rem // OW
    ow = rem % OW

    # Accumulator [BLOCK_SP, OC_C]
    acc = tl.zeros((BLOCK_SP, OC_C), dtype=tl.float32)

    # Input base for this batch
    x_batch = x_ptr + pid_n * IC * ID * IH * IW

    # Loop over kernel and IC; K=IC*KD*KH*KW
    for ic in tl.static_range(IC_C):
        for kd in tl.static_range(KD):
            for kh in tl.static_range(KH):
                for kw in tl.static_range(KW):
                    id_ = od + kd
                    ih = oh + kh
                    iw = ow + kw
                    # input offset
                    x_off = ic * (ID * IH * IW) + id_ * (IH * IW) + ih * IW + iw
                    x_val = tl.load(x_batch + x_off, mask=sp_mask, other=0.0)
                    # weight offsets for all OC at this (ic, kd, kh, kw)
                    # weight layout: [OC, IC, KD, KH, KW]
                    oc_off = tl.arange(0, OC_C)
                    w_off = oc_off * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)
                    acc += x_val[:, None] * w_val[None, :]

    # Add conv bias [OC]
    cb = tl.load(cb_ptr + tl.arange(0, OC_C))
    acc += cb[None, :]

    # Scaling factor and bias [OC]
    scale = tl.load(scale_ptr + tl.arange(0, OC_C))
    bias = tl.load(bias_ptr + tl.arange(0, OC_C))

    y = acc * scale[None, :]
    # tanh
    y = 2.0 * tl.sigmoid(2.0 * y) - 1.0
    y = y * bias[None, :]
    y = tl.sigmoid(y)

    # Store: output layout [N, OC, OD, OH, OW]
    # for each oc in 0..OC_C, store BLOCK_SP elements
    out_batch = out_ptr + pid_n * OC * ODHW
    for oc in tl.static_range(OC_C):
        out_off = oc * ODHW + sp_off
        tl.store(out_batch + out_off, y[:, oc], mask=sp_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        w = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()
        scale = self.scaling_factor.contiguous().view(-1)
        bias = self.bias.contiguous().view(-1)

        BLOCK_SP = 128
        ODHW = OD * OH * OW
        grid = (N, (ODHW + BLOCK_SP - 1) // BLOCK_SP)

        conv3d_fused_kernel[grid](
            x, w, cb, scale, bias, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD=KD, KH=KH, KW=KW,
            BLOCK_SP=BLOCK_SP,
            OC_C=OC,
            IC_C=IC,
            num_warps=4,
            num_stages=2,
        )
        return out