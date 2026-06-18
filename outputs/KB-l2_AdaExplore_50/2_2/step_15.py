import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_scatter_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # program ids: (n, ih, iw)
    pid = tl.program_id(0)
    n = tl.program_id(1)
    ih = pid // IW
    iw = pid % IW

    offs_oc = tl.arange(0, BLOCK_OC)
    offs_ic = tl.arange(0, BLOCK_IC)

    # Load input vector x[n, :, ih, iw] of length IC
    # accumulator over output channels
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # We'll just loop over IC and accumulate per kh,kw
    # Actually better: for each (kh,kw), compute acc[oc] = sum_ic x[ic]*w[ic,oc,kh,kw]
    # Layout: weight shape (IC, OC, KH, KW)
    # We do one kernel computing the full IC reduction for one (n, ih, iw),
    # producing KH*KW outputs each of size OC, and atomic_add to output.

    x_base = x_ptr + n * IC * IH * IW + ih * IW + iw
    # x[ic] = x_base + ic*IH*IW
    x_offs = offs_ic * (IH * IW)
    mask_ic = offs_ic < IC

    mask_oc = offs_oc < OC

    for kh in tl.static_range(0, KH):
        oh = ih * SH - PH + kh
        h_valid = (oh >= 0) & (oh < OH)
        for kw in tl.static_range(0, KW):
            ow = iw * SW - PW + kw
            w_valid = h_valid & (ow >= 0) & (ow < OW)

            # accumulate acc = sum_ic x[ic] * w[ic, :, kh, kw]
            acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
            # Loop over IC in tiles
            for ic_start in range(0, IC, BLOCK_IC):
                ic_idx = ic_start + offs_ic
                m_ic = ic_idx < IC
                x_vals = tl.load(x_base + ic_idx * (IH * IW), mask=m_ic, other=0.0)
                # weight pointer w[ic, oc, kh, kw]: w_ptr + ic*OC*KH*KW + oc*KH*KW + kh*KW + kw
                w_offs = ic_idx[:, None] * (OC * KH * KW) + offs_oc[None, :] * (KH * KW) + kh * KW + kw
                w_mask = m_ic[:, None] & mask_oc[None, :]
                w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)
                acc += tl.sum(x_vals[:, None] * w_vals, axis=0)

            # scatter-add to output[n, :, oh, ow]
            out_offs = n * OC * OH * OW + offs_oc * (OH * OW) + oh * OW + ow
            store_mask = mask_oc & w_valid
            tl.atomic_add(out_ptr + out_offs, acc, mask=store_mask)


@triton.jit
def epilogue_kernel(
    out_ptr, bias_ptr,
    N, OC, OH, OW,
    inv_scale,  # 1/scaling_factor
    scale,       # scaling_factor
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * OC * OH * OW
    mask = offs < total

    x = tl.load(out_ptr + offs, mask=mask, other=0.0)
    # compute oc index
    oc_idx = (offs // (OH * OW)) % OC
    b = tl.load(bias_ptr + oc_idx, mask=mask, other=0.0)

    x = x + b
    x = tl.minimum(tl.maximum(x, 0.0), 1.0)
    x = x * scale
    x = tl.minimum(tl.maximum(x, 0.0), 1.0)
    x = x * inv_scale

    tl.store(out_ptr + offs, x, mask=mask)


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

        # mirror nn.ConvTranspose2d parameter shapes
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OPH = OPW = self.output_padding

        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        # output buffer initialized to conv_transpose bias broadcast
        out = torch.empty((N, OC, OH, OW), dtype=x.dtype, device=x.device)
        # initialize with conv bias
        ct_bias = self.conv_transpose.bias
        if ct_bias is not None:
            out.copy_(ct_bias.view(1, OC, 1, 1).expand(N, OC, OH, OW))
        else:
            out.zero_()

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KH, KW)

        BLOCK_OC = triton.next_power_of_2(OC)
        BLOCK_IC = min(64, triton.next_power_of_2(IC))

        grid = (IH * IW, N)
        conv_transpose_scatter_kernel[grid](
            x, weight, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW, SH, SW, PH, PW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_IC=BLOCK_IC,
            num_warps=4,
        )

        # epilogue: + bias, clamp, scale, clamp, /scale
        total = N * OC * OH * OW
        BLOCK = 1024
        grid2 = (triton.cdiv(total, BLOCK),)
        epilogue_kernel[grid2](
            out, self.bias.contiguous().view(-1),
            N, OC, OH, OW,
            1.0 / self.scaling_factor,
            float(self.scaling_factor),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out