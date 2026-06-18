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
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # one program per (n, ih, iw)
    pid = tl.program_id(0)
    iw = pid % IW
    ih = (pid // IW) % IH
    n = pid // (IH * IW)

    # output base position (top-left of the kernel footprint)
    oh_base = ih * STRIDE_H - PAD_H
    ow_base = iw * STRIDE_W - PAD_W

    offs_oc = tl.arange(0, BLOCK_OC)
    offs_ic = tl.arange(0, BLOCK_IC)

    # load input vector x[n, :, ih, iw] of length IC
    ic_mask = offs_ic < IC
    x_off = ((n * IC) + offs_ic) * (IH * IW) + ih * IW + iw
    x_vec = tl.load(x_ptr + x_off, mask=ic_mask, other=0.0)  # [BLOCK_IC]

    oc_mask = offs_oc < OC

    # loop over kernel positions
    for kh in range(0, KH):
        oh = oh_base + kh
        oh_valid = (oh >= 0) & (oh < OH)
        for kw in range(0, KW):
            ow = ow_base + kw
            ow_valid = (ow >= 0) & (ow < OW)
            valid = oh_valid & ow_valid
            # weight slice: w[ic, oc, kh, kw], shape [BLOCK_IC, BLOCK_OC]
            w_off = (offs_ic[:, None] * OC * KH * KW
                     + offs_oc[None, :] * KH * KW
                     + kh * KW + kw)
            w_mask = ic_mask[:, None] & oc_mask[None, :]
            w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]
            # contribution: x_vec[:, None] * w_vals -> sum over ic -> [BLOCK_OC]
            contrib = tl.sum(x_vec[:, None] * w_vals, axis=0)  # [BLOCK_OC]

            out_off = ((n * OC + offs_oc) * OH + oh) * OW + ow
            store_mask = oc_mask & valid
            # atomic add
            tl.atomic_add(out_ptr + out_off, contrib, mask=store_mask)


@triton.jit
def _bias_tanh_kernel(
    out_ptr, convbias_ptr, bias_ptr,
    N, OC, OH, OW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * OC * OH * OW
    mask = offs < total

    # compute oc index
    spatial = OH * OW
    oc = (offs // spatial) % OC

    val = tl.load(out_ptr + offs, mask=mask, other=0.0)
    cb = tl.load(convbias_ptr + oc, mask=mask, other=0.0)
    b = tl.load(bias_ptr + oc, mask=mask, other=0.0)
    val = val + cb - b
    # tanh
    e2 = tl.exp(2.0 * val)
    out = (e2 - 1.0) / (e2 + 1.0)
    tl.store(out_ptr + offs, out, mask=mask)


def conv_transpose2d_triton(x, weight, conv_bias, extra_bias,
                            stride, padding, output_padding):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    sh, sw = stride
    ph, pw = padding
    oph, opw = output_padding

    OH = (IH - 1) * sh - 2 * ph + KH + oph
    OW = (IW - 1) * sw - 2 * pw + KW + opw

    x = x.contiguous()
    weight = weight.contiguous()

    out = torch.zeros((N, OC, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_OC = triton.next_power_of_2(OC)
    BLOCK_IC = triton.next_power_of_2(IC)

    grid = (N * IH * IW,)
    _scatter_kernel[grid](
        x, weight, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        sh, sw,
        ph, pw,
        BLOCK_OC=BLOCK_OC,
        BLOCK_IC=BLOCK_IC,
        num_warps=4,
    )

    # fuse conv bias + extra bias + tanh
    extra_bias_flat = extra_bias.reshape(-1).contiguous()
    conv_bias = conv_bias.contiguous()
    total = N * OC * OH * OW
    BLOCK = 1024
    grid2 = (triton.cdiv(total, BLOCK),)
    _bias_tanh_kernel[grid2](
        out, conv_bias, extra_bias_flat,
        N, OC, OH, OW,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape,
                 stride=2, padding=1, output_padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding

    def forward(self, x):
        x = x.cuda().contiguous()
        weight = self.conv_transpose.weight
        conv_bias = self.conv_transpose.bias
        if conv_bias is None:
            conv_bias = torch.zeros(weight.shape[1], device=x.device, dtype=x.dtype)
        return conv_transpose2d_triton(
            x, weight, conv_bias, self.bias,
            self.stride, self.padding, self.output_padding
        )