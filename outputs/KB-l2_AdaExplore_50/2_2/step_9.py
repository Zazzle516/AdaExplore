import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    inv_scale,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]

    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    # Initialize accumulator
    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    # Loop over kernel positions
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # ih_num = oh + PAD - kh; valid if divisible by stride
            ih_num = oh + PAD - kh  # [BLOCK_HW]
            iw_num = ow + PAD - kw
            ih = ih_num // STRIDE
            iw = iw_num // STRIDE
            valid_div = ((ih_num % STRIDE) == 0) & ((iw_num % STRIDE) == 0)
            valid_range = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            spatial_valid = valid_div & valid_range & hw_mask  # [BLOCK_HW]

            # Loop over input channels
            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                # Load x[N, IC, IH, IW] -> shape [BLOCK_HW, BLOCK_IC]
                x_ptrs = (
                    x_ptr
                    + pid_n * (IC * IH * IW)
                    + ic_offs[None, :] * (IH * IW)
                    + ih[:, None] * IW
                    + iw[:, None]
                )
                x_mask = spatial_valid[:, None] & ic_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_HW, BLOCK_IC]

                # Load weight[IC, OC, KH, KW] -> shape [BLOCK_IC, BLOCK_OC]
                w_ptrs = (
                    w_ptr
                    + ic_offs[:, None] * (OC * KH * KW)
                    + oc_offs[None, :] * (KH * KW)
                    + kh * KW
                    + kw
                )
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]

                acc += tl.dot(x_vals, w_vals)

    # Add bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + b_vals[None, :]

    # Fused epilogue: clamp(0,1), scale, clamp(0,1), /scale
    # First clamp 0..1
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    # Multiply by scale, clamp, divide by scale - equivalent to clamp(acc, 0, 1/scale)
    # acc * s, clamp 0..1, then /s -> clamp(acc, 0, 1/s)
    acc = tl.minimum(acc, inv_scale)

    # Store: out[N, OC, OH, OW]
    out_ptrs = (
        out_ptr
        + pid_n * (OC * OH * OW)
        + oc_offs[None, :] * (OH * OW)
        + hw_offs[:, None]
    )
    out_mask = oc_mask[None, :] & hw_mask[:, None]
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        KH = self.kernel_size
        KW = self.kernel_size
        OC = self.out_channels
        stride = self.stride
        pad = self.padding
        out_pad = self.output_padding

        OH = (IH - 1) * stride - 2 * pad + KH + out_pad
        OW = (IW - 1) * stride - 2 * pad + KW + out_pad

        # Combine conv bias and self.bias
        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        conv_bias = self.conv_transpose.bias  # [OC] or None
        extra_bias = self.bias.view(-1)  # [OC]
        if conv_bias is not None:
            total_bias = (conv_bias + extra_bias).contiguous()
        else:
            total_bias = extra_bias.contiguous()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_HW = 64
        BLOCK_IC = 32

        grid = (
            N,
            triton.cdiv(OC, BLOCK_OC),
            triton.cdiv(OH * OW, BLOCK_HW),
        )

        inv_scale = 1.0 / self.scaling_factor

        conv_transpose2d_kernel[grid](
            x, weight, total_bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            stride, pad,
            inv_scale,
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            BLOCK_IC=BLOCK_IC,
            num_warps=4,
            num_stages=2,
        )

        return out