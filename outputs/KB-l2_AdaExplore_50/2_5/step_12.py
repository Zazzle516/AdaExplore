import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv_transpose_gather_kernel(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [IC, OC, KH, KW]
    cb_ptr,          # [OC] convolution bias
    sb_ptr,          # [OC] subtract bias
    out_ptr,         # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,  # spatial tile
    BLOCK_N: tl.constexpr,  # OC tile (== OC = 64)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)  # batch

    # offsets in spatial domain
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    spatial = OH * OW
    mask_m = offs_m < spatial
    mask_n = offs_n < OC

    oh = offs_m // OW
    ow = offs_m % OW

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For each kernel position, find the input position that contributes
    # i = (o + pad - k) / stride, valid if (o + pad - k) % stride == 0
    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        ih_valid = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)

        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW)

            valid_m = ih_valid & iw_valid & mask_m  # [BLOCK_M]

            # Load weight tile [IC, BLOCK_N] for this (kh, kw)
            # weight layout: [IC, OC, KH, KW]; offset for (ic, oc, kh, kw)
            # = ic*OC*KH*KW + oc*KH*KW + kh*KW + kw
            # We'll load full IC inside the kernel via a loop
            for ic in range(0, IC):
                # load x[b, ic, ih, iw] for each m
                x_off = (pid_b * IC + ic) * IH * IW + ih * IW + iw
                x_val = tl.load(x_ptr + x_off, mask=valid_m, other=0.0)  # [BLOCK_M]

                # load weight[ic, :, kh, kw] -> [OC]
                w_off = ic * OC * KH * KW + offs_n * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=mask_n, other=0.0)  # [BLOCK_N]

                acc += x_val[:, None] * w_val[None, :]

    # add conv bias, subtract sub_bias, tanh
    cb = tl.load(cb_ptr + offs_n, mask=mask_n, other=0.0)
    sb = tl.load(sb_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + cb[None, :] - sb[None, :]

    # tanh
    e2 = tl.exp(2.0 * acc)
    res = (e2 - 1.0) / (e2 + 1.0)

    # store: out[b, oc, oh, ow]
    out_off = (pid_b * OC + offs_n[None, :]) * spatial + offs_m[:, None]
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, res, mask=store_mask)


@triton.jit
def _conv_transpose_gather_kernel_v2(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [IC, OC, KH, KW]
    cb_ptr,          # [OC]
    sb_ptr,          # [OC]
    out_ptr,         # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_ic = tl.arange(0, BLOCK_IC)

    spatial = OH * OW
    mask_m = offs_m < spatial
    mask_n = offs_n < OC

    oh = offs_m // OW
    ow = offs_m % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        ih_valid = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)

        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW)

            valid_m = ih_valid & iw_valid & mask_m

            for ic_start in range(0, IC, BLOCK_IC):
                ic_idx = ic_start + offs_ic
                ic_mask = ic_idx < IC

                # x[b, ic_idx, ih, iw]: shape [BLOCK_M, BLOCK_IC]
                x_off = (pid_b * IC + ic_idx[None, :]) * (IH * IW) + ih[:, None] * IW + iw[:, None]
                x_mask = valid_m[:, None] & ic_mask[None, :]
                x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                # w[ic_idx, oc_idx, kh, kw]: shape [BLOCK_IC, BLOCK_N]
                w_off = ic_idx[:, None] * (OC * KH * KW) + offs_n[None, :] * (KH * KW) + kh * KW + kw
                w_mask = ic_mask[:, None] & mask_n[None, :]
                w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                acc += tl.dot(x_tile, w_tile)

    cb = tl.load(cb_ptr + offs_n, mask=mask_n, other=0.0)
    sb = tl.load(sb_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + cb[None, :] - sb[None, :]

    e2 = tl.exp(2.0 * acc)
    res = (e2 - 1.0) / (e2 + 1.0)

    out_off = (pid_b * OC + offs_n[None, :]) * spatial + offs_m[:, None]
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, res, mask=store_mask)


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
        OC = self.out_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        OP = self.output_padding

        OH = (IH - 1) * S - 2 * P + KH + OP
        OW = (IW - 1) * S - 2 * P + KW + OP

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        cbias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        sbias = self.bias.view(-1).contiguous()

        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_IC = 32

        spatial = OH * OW
        grid = (
            (spatial + BLOCK_M - 1) // BLOCK_M,
            (OC + BLOCK_N - 1) // BLOCK_N,
            N,
        )

        _conv_transpose_gather_kernel_v2[grid](
            x, weight, cbias, sbias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH=KH, KW=KW,
            STRIDE=S, PAD=P,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_IC=BLOCK_IC,
            num_warps=4, num_stages=2,
        )

        return out