import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_scatter_kernel(
    x_ptr,           # [N, IC, H, W]
    w_ptr,           # [IC, OC, KH, KW]
    out_ptr,         # [N, OC, OH, OW]
    N, IC, OC, H, W, OH, OW, KH, KW,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # program id layout: (n, ic, hw_block, oc_block)
    pid_n = tl.program_id(0)
    pid_ic = tl.program_id(1)
    pid_hw = tl.program_id(2)
    # second grid dim packs oc_block via pid_ic? No — use 3D grid.
    # We'll use n in pid0, hw_block in pid1, oc_block in pid2 and loop over IC.
    # Actually let's restructure below.
    pass


@triton.jit
def conv_transpose_kernel(
    x_ptr,           # [N, IC, H, W]
    w_ptr,           # [IC, OC, KH, KW]
    b_conv_ptr,      # [OC]
    bias_ptr,        # [OC]
    out_ptr,         # [N, OC, OH, OW]
    N, IC, OC, H, W, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    scaling_factor,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_oc = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    oh = offs_hw // OW
    ow = offs_hw % OW
    mask_hw = offs_hw < (OH * OW)
    mask_oc = offs_oc < OC

    # accumulator
    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    # For each (kh, kw), the corresponding input position is:
    # ih_num = oh + PAD_H - kh ; valid if ih_num % STRIDE_H == 0 and 0 <= ih_num/STRIDE_H < H
    # iw similarly.

    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD_H - kh
        ih = ih_num // STRIDE_H
        ih_valid = (ih_num >= 0) & (ih_num < H * STRIDE_H) & ((ih_num % STRIDE_H) == 0) & (ih >= 0) & (ih < H)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD_W - kw
            iw = iw_num // STRIDE_W
            iw_valid = (iw_num >= 0) & (iw_num < W * STRIDE_W) & ((iw_num % STRIDE_W) == 0) & (iw >= 0) & (iw < W)
            valid = ih_valid & iw_valid & mask_hw  # [BLOCK_HW]

            # Loop over IC
            for ic in range(0, IC):
                # Load x[n, ic, ih, iw] for each hw element
                x_offset = pid_n * (IC * H * W) + ic * (H * W) + ih * W + iw
                x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)  # [BLOCK_HW]

                # Load w[ic, :, kh, kw] for oc block
                w_offset = ic * (OC * KH * KW) + offs_oc * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_offset, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                acc += x_val[:, None] * w_val[None, :]

    # add conv bias
    b_conv = tl.load(b_conv_ptr + offs_oc, mask=mask_oc, other=0.0)  # [BLOCK_OC]
    acc += b_conv[None, :]

    # add extra bias (per OC)
    bias_val = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)  # [BLOCK_OC]
    acc += bias_val[None, :]

    # clamp [0, 1]
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    # scale
    acc = acc * scaling_factor
    # clamp [0, 1]
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    # divide
    acc = acc / scaling_factor

    # store: out[n, oc, oh, ow]
    out_offset = pid_n * (OC * OH * OW) + offs_oc[None, :] * (OH * OW) + offs_hw[:, None]
    out_mask = mask_hw[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


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

        # Use nn.ConvTranspose2d to get standard initialization for weight and bias
        ct = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                 stride=stride, padding=padding,
                                 output_padding=output_padding)
        # weight shape: [in_channels, out_channels, KH, KW]
        self.weight = nn.Parameter(ct.weight.detach().clone())
        self.conv_bias = nn.Parameter(ct.bias.detach().clone())
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        STRIDE_H = self.stride
        STRIDE_W = self.stride
        PAD_H = self.padding
        PAD_W = self.padding
        OUT_PAD = self.output_padding

        OH = (H - 1) * STRIDE_H - 2 * PAD_H + KH + OUT_PAD
        OW = (W - 1) * STRIDE_W - 2 * PAD_W + KW + OUT_PAD

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        bias_flat = self.bias.view(-1).contiguous()
        weight = self.weight.contiguous()
        conv_bias = self.conv_bias.contiguous()

        BLOCK_OC = 64
        BLOCK_HW = 64

        grid = (
            N,
            triton.cdiv(OH * OW, BLOCK_HW),
            triton.cdiv(OC, BLOCK_OC),
        )

        conv_transpose_kernel[grid](
            x, weight, conv_bias, bias_flat, out,
            N, IC, OC, H, W, OH, OW,
            KH, KW,
            STRIDE_H, STRIDE_W,
            PAD_H, PAD_W,
            float(self.scaling_factor),
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=2,
        )

        return out