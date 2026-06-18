import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _scatter_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    BLOCK_IC: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, ih, iw) -> processes all OC outputs touched
    pid = tl.program_id(0)
    n = tl.program_id(1)

    ih = pid // IW
    iw = pid % IW

    # Load input vector x[n, :, ih, iw] of shape (IC,)
    ic_offs = tl.arange(0, BLOCK_IC)
    ic_mask = ic_offs < IC
    x_offset = n * (IC * IH * IW) + ic_offs * (IH * IW) + ih * IW + iw
    x_vec = tl.load(x_ptr + x_offset, mask=ic_mask, other=0.0)  # (BLOCK_IC,)

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # For each kernel position, compute output coords and scatter-add
    for kh in range(0, KH):
        oh = ih * STRIDE_H - PAD_H + kh
        for kw in range(0, KW):
            ow = iw * STRIDE_W - PAD_W + kw
            valid = (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW)
            if valid:
                # weight shape (IC, OC, KH, KW), layout: ic*OC*KH*KW + oc*KH*KW + kh*KW + kw
                w_off = ic_offs[:, None] * (OC * KH * KW) + oc_offs[None, :] * (KH * KW) + kh * KW + kw
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # (BLOCK_IC, BLOCK_OC)
                # contribution: sum over ic of x_vec[ic] * w_tile[ic, oc]
                contrib = tl.sum(x_vec[:, None] * w_tile, axis=0)  # (BLOCK_OC,)
                out_off = n * (OC * OH * OW) + oc_offs * (OH * OW) + oh * OW + ow
                tl.atomic_add(out_ptr + out_off, contrib, mask=oc_mask)


@triton.jit
def _bias_tanh_kernel(
    out_ptr, conv_bias_ptr, sub_bias_ptr,
    N, OC, OH, OW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * OC * OH * OW
    mask = offs < total

    spatial = OH * OW
    oc = (offs // spatial) % OC

    v = tl.load(out_ptr + offs, mask=mask, other=0.0)
    cb = tl.load(conv_bias_ptr + oc, mask=mask, other=0.0)
    sb = tl.load(sub_bias_ptr + oc, mask=mask, other=0.0)
    z = v + cb - sb
    # tanh via exp
    e2 = tl.exp(2.0 * z)
    res = (e2 - 1.0) / (e2 + 1.0)
    tl.store(out_ptr + offs, res, mask=mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
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
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        OC = self.out_channels
        SH = SW = self.stride
        PH = PW = self.padding
        OPH = OPW = self.output_padding

        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KH, KW)

        out = torch.zeros((N, OC, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_IC = _next_pow2(IC)
        BLOCK_OC = _next_pow2(OC)

        grid = (IH * IW, N)
        _scatter_kernel[grid](
            x, weight, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            SH, SW,
            PH, PW,
            BLOCK_IC=BLOCK_IC,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
        )

        conv_bias = self.conv_transpose.bias.contiguous()
        sub_bias = self.bias.view(-1).contiguous()

        total = N * OC * OH * OW
        BLOCK = 1024
        grid2 = ((total + BLOCK - 1) // BLOCK,)
        _bias_tanh_kernel[grid2](
            out, conv_bias, sub_bias,
            N, OC, OH, OW,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out