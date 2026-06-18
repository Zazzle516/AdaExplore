import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, extra_bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    OHW = OH * OW
    S_total = OD * OH * OW

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)      # [BLOCK_S]

    oc_mask = offs_oc < OC
    s_mask = offs_s < S_total

    od = offs_s // OHW
    rem = offs_s % OHW
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BLOCK_S, BLOCK_OC), dtype=tl.float32)

    # Loop over kernel positions
    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_q = id_num // STRIDE
        id_r = id_num - id_q * STRIDE
        valid_d = (id_r == 0) & (id_q >= 0) & (id_q < ID)

        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_q = ih_num // STRIDE
            ih_r = ih_num - ih_q * STRIDE
            valid_h = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)

            for kw in tl.static_range(0, KW):
                iw_num = ow + PAD - kw
                iw_q = iw_num // STRIDE
                iw_r = iw_num - iw_q * STRIDE
                valid_w = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW)

                valid = valid_d & valid_h & valid_w & s_mask  # [BLOCK_S]

                # Input offset base (without IC): n*IC*ID*IH*IW + id*IH*IW + ih*IW + iw
                in_spatial = id_q * (IH * IW) + ih_q * IW + iw_q  # [BLOCK_S]
                in_base = pid_n * (IC * ID * IH * IW) + in_spatial  # [BLOCK_S]

                # Weight base (without IC): w[ic, oc, kd, kh, kw] -> ic*OC*KD*KH*KW + oc*KD*KH*KW + kd*KH*KW + kh*KW + kw
                w_kpos = kd * (KH * KW) + kh * KW + kw
                w_base = offs_oc * (KD * KH * KW) + w_kpos  # [BLOCK_OC]

                # Accumulate over IC
                for ic in range(0, IC):
                    in_off = in_base + ic * (ID * IH * IW)  # [BLOCK_S]
                    w_off = ic * (OC * KD * KH * KW) + w_base  # [BLOCK_OC]

                    x_val = tl.load(x_ptr + in_off, mask=valid, other=0.0)  # [BLOCK_S]
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += x_val[:, None] * w_val[None, :]

    # Add conv bias
    cb = tl.load(conv_bias_ptr + offs_oc, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    conv_out = acc + cb[None, :]  # [BLOCK_S, BLOCK_OC]

    # Extra bias (per channel)
    eb = tl.load(extra_bias_ptr + offs_oc, mask=oc_mask, other=0.0)  # [BLOCK_OC]

    # Fused epilogue: (2x + bias) * x + x
    res = (2.0 * conv_out + eb[None, :]) * conv_out + conv_out

    # Store: out shape is (N, OC, OD, OH, OW)
    out_off = pid_n * (OC * S_total) + offs_oc[None, :] * S_total + offs_s[:, None]
    store_mask = oc_mask[None, :] & s_mask[:, None]
    tl.store(out_ptr + out_off, res, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        conv_bias = self.conv_transpose.bias.contiguous()
        extra_bias = self.bias.contiguous().view(-1)

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        OP = self.output_padding

        OD = (ID - 1) * S - 2 * P + KD + OP
        OH = (IH - 1) * S - 2 * P + KH + OP
        OW = (IW - 1) * S - 2 * P + KW + OP

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_S = 128

        S_total = OD * OH * OW
        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (S_total + BLOCK_S - 1) // BLOCK_S)

        conv_transpose3d_fused_kernel[grid](
            x, weight, conv_bias, extra_bias, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            S, P,
            BLOCK_OC=BLOCK_OC,
            BLOCK_S=BLOCK_S,
            num_warps=4,
            num_stages=2,
        )

        return out