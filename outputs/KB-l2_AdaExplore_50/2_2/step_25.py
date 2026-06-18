import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'OC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv_transpose2d_kernel(
    x_ptr,            # [N, IC, IH, IW]
    w_ptr,            # [IC, OC, KH, KW]  (PyTorch ConvTranspose2d weight layout)
    b_ptr,            # [OC] (combined bias = conv_bias + extra_bias)
    out_ptr,          # [N, OC, OH, OW]
    clamp_max,        # scalar: min(1.0, 1.0/scaling_factor)
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_M: tl.constexpr,   # tile over OC
    BLOCK_N: tl.constexpr,   # tile over OH*OW (spatial)
):
    pid_n = tl.program_id(0)         # batch index
    pid_oc = tl.program_id(1)        # OC tile
    pid_sp = tl.program_id(2)        # spatial tile

    offs_oc = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_sp = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    mask_oc = offs_oc < OC
    mask_sp = offs_sp < (OH * OW)

    oh = offs_sp // OW                # [BLOCK_N]
    ow = offs_sp % OW                 # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kernel positions and input channels
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # For ConvTranspose2d: output(oh, ow) gets contribution from
            # input(ih, iw) where ih*stride - pad + kh = oh
            #   => ih = (oh + pad - kh) / stride, must be integer and in range
            ih_num = oh + PAD_H - kh
            iw_num = ow + PAD_W - kw

            ih = ih_num // STRIDE_H
            iw = iw_num // STRIDE_W

            valid = ((ih_num - ih * STRIDE_H) == 0) & \
                    ((iw_num - iw * STRIDE_W) == 0) & \
                    (ih >= 0) & (ih < IH) & \
                    (iw >= 0) & (iw < IW)        # [BLOCK_N]

            valid_sp = valid & mask_sp

            # Loop over input channels
            for ic in range(0, IC):
                # Load input x[n, ic, ih, iw] : [BLOCK_N]
                x_offs = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                x_vals = tl.load(x_ptr + x_offs, mask=valid_sp, other=0.0)  # [BLOCK_N]

                # Load weight w[ic, oc, kh, kw] : [BLOCK_M]
                w_offs = ic * (OC * KH * KW) + offs_oc * (KH * KW) + kh * KW + kw
                w_vals = tl.load(w_ptr + w_offs, mask=mask_oc, other=0.0)  # [BLOCK_M]

                acc += w_vals[:, None] * x_vals[None, :]

    # Add bias
    bias_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)  # [BLOCK_M]
    acc += bias_vals[:, None]

    # Epilogue: clamp(x, 0, clamp_max)
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, clamp_max)

    # Store: out[n, oc, oh, ow]
    out_offs = pid_n * (OC * OH * OW) + offs_oc[:, None] * (OH * OW) + offs_sp[None, :]
    mask_out = mask_oc[:, None] & mask_sp[None, :]
    tl.store(out_ptr + out_offs, acc, mask=mask_out)


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

        # Create the same nn.ConvTranspose2d to inherit identical param init
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        KH = self.kernel_size
        KW = self.kernel_size
        OC = self.out_channels

        # Output spatial size for ConvTranspose2d
        OH = (IH - 1) * self.stride - 2 * self.padding + KH + self.output_padding
        OW = (IW - 1) * self.stride - 2 * self.padding + KW + self.output_padding

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        conv_bias = self.conv_transpose.bias              # [OC]
        extra_bias = self.bias.view(-1)                   # [OC]
        combined_bias = (conv_bias + extra_bias).contiguous()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        s = self.scaling_factor
        clamp_max = min(1.0, 1.0 / s)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(OH * OW, meta['BLOCK_N']),
        )

        conv_transpose2d_kernel[grid](
            x, weight, combined_bias, out,
            clamp_max,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            self.stride, self.stride,
            self.padding, self.padding,
        )

        return out