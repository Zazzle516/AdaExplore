import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [IC, OC, KH, KW]
    b_ptr,           # [OC] (conv bias)
    bias2_ptr,       # [OC] (extra bias)
    out_ptr,         # [N, OC, OH, OW]
    N, IC, OC,
    IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    SCALE: tl.constexpr, INV_SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oh = sp_offs // OW
    ow = sp_offs % OW
    sp_mask = sp_offs < (OH * OW)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For each kernel position, find valid input position
    # ih = (oh + pad - kh) / stride if divisible
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_num = oh + PAD - kh
            iw_num = ow + PAD - kw
            ih = ih_num // STRIDE
            iw = iw_num // STRIDE
            valid = ((ih_num % STRIDE) == 0) & ((iw_num % STRIDE) == 0) \
                    & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & sp_mask
            # accumulate over IC
            for ic in range(0, IC):
                # x[pid_n, ic, ih, iw]
                x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)  # [BLOCK_SP]
                # w[ic, oc_offs, kh, kw]
                w_off = ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
                wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                acc += xv[:, None] * wv[None, :]

    # add conv bias
    bv = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc += bv[None, :]
    # add extra bias
    b2 = tl.load(bias2_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b2[None, :]

    # clamp 0..1, scale, clamp 0..1, divide
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * SCALE
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * INV_SCALE

    # store [BLOCK_SP, BLOCK_OC] into out [N, OC, OH, OW]
    out_off = pid_n * (OC * OH * OW) + oc_offs[None, :] * (OH * OW) + sp_offs[:, None]
    mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor

        # Match nn.ConvTranspose2d default init
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv_transpose.weight.contiguous().cuda()  # [IC, OC, KH, KW]
        cb = self.conv_transpose.bias.contiguous().cuda()   # [OC]
        b2 = self.bias.view(-1).contiguous().cuda()         # [OC]

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        OP = self.output_padding

        OH = (IH - 1) * S - 2 * P + KH + OP
        OW = (IW - 1) * S - 2 * P + KW + OP

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 128

        grid = (
            N,
            triton.cdiv(OC, BLOCK_OC),
            triton.cdiv(OH * OW, BLOCK_SP),
        )

        conv_transpose2d_kernel[grid](
            x, w, cb, b2, out,
            N, IC, OC,
            IH, IW, OH, OW,
            KH, KW,
            S, P,
            float(self.scaling_factor), float(1.0 / self.scaling_factor),
            BLOCK_OC, BLOCK_SP,
            num_warps=4, num_stages=2,
        )

        return out