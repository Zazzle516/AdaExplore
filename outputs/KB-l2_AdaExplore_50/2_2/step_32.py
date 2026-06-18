import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, extra_bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PADDING: tl.constexpr,
    INV_SCALE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Program ids
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    hw_start = pid_hw * BLOCK_HW
    offs = hw_start + tl.arange(0, BLOCK_HW)
    oh = offs // OW
    ow = offs % OW
    mask = offs < (OH * OW)

    # accumulator
    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    # For each kernel position (kh, kw):
    #   we need: (oh + PADDING - kh) divisible by STRIDE, ih = (oh+PADDING-kh)/STRIDE in [0, IH)
    # ConvTranspose2d: output[n,oc,oh,ow] = sum_{ic,kh,kw} input[n,ic,ih,iw] * weight[ic,oc,kh,kw]
    # where oh = ih*STRIDE - PADDING + kh => ih = (oh + PADDING - kh) / STRIDE

    for kh in tl.static_range(0, KH):
        h_num = oh + PADDING - kh
        ih = h_num // STRIDE
        h_valid = ((h_num % STRIDE) == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            w_num = ow + PADDING - kw
            iw = w_num // STRIDE
            w_valid = ((w_num % STRIDE) == 0) & (iw >= 0) & (iw < IW)
            valid = h_valid & w_valid & mask

            # iterate over ic
            for ic in range(0, IC):
                # x[n, ic, ih, iw]
                x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                # w[ic, oc, kh, kw]
                w_off = ic * (OC * KH * KW) + pid_oc * (KH * KW) + kh * KW + kw
                wv = tl.load(w_ptr + w_off)
                acc += xv * wv

    # Add conv bias
    cb = tl.load(conv_bias_ptr + pid_oc)
    acc = acc + cb
    # Add extra bias (shape OC,1,1)
    eb = tl.load(extra_bias_ptr + pid_oc)
    acc = acc + eb
    # clamp [0,1], * scale, clamp [0,1], / scale  --> equivalent to clamp(acc, 0, INV_SCALE) (since *scale then clamp[0,1] then /scale = clamp(acc, 0, 1/scale)) but we also have the initial clamp[0,1].
    # The chain: y = clamp(acc,0,1); y = y*S; y = clamp(y,0,1); y = y/S
    # = clamp(clamp(acc,0,1)*S, 0, 1) / S
    # = clamp(acc*S, 0, 1)/S  when acc>=0 (clamp(acc,0,1)*S = min(acc,1)*S; if acc<=1, =acc*S; clamped to 1 -> min(acc*S,1); if acc>1, =S, clamped to 1 -> 1; same as min(acc*S,1) when acc>=0). For acc<0, clamp(acc,0,1)=0, result 0; and clamp(acc*S,0,1)=0 too.
    # So result = clamp(acc*S, 0, 1) / S = clamp(acc, 0, 1/S)
    inv_s = INV_SCALE
    acc = tl.minimum(tl.maximum(acc, 0.0), inv_s)

    # store
    out_off = pid_n * (OC * OH * OW) + pid_oc * (OH * OW) + offs
    tl.store(out_ptr + out_off, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor

        # Mimic nn.ConvTranspose2d initialization
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   output_padding=output_padding)
        self.conv_weight = nn.Parameter(conv.weight.detach().clone())
        self.conv_bias = nn.Parameter(conv.bias.detach().clone())
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        OP = self.output_padding

        OH = (IH - 1) * S - 2 * P + KH + OP
        OW = (IW - 1) * S - 2 * P + KW + OP

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        w = self.conv_weight.contiguous()
        cb = self.conv_bias.contiguous()
        eb = self.bias.contiguous().view(-1)

        BLOCK_HW = 128
        grid = (N, OC, triton.cdiv(OH * OW, BLOCK_HW))
        inv_scale = 1.0 / float(self.scaling_factor)

        conv_transpose2d_fused_kernel[grid](
            x, w, cb, eb, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            S, P,
            inv_scale,
            BLOCK_HW,
            num_warps=4,
            num_stages=2,
        )
        return out