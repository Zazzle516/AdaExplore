import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PADDING: tl.constexpr,
    ADD_VALUE: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_oc = tl.program_id(2)

    num_hw_blocks = tl.cdiv(OH * OW, BLOCK_HW)
    pid_hw = pid % num_hw_blocks

    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    oh = hw_offsets // OW
    ow = hw_offsets % OW
    hw_mask = hw_offsets < (OH * OW)

    oc_offsets = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < OC

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    # For ConvTranspose2d gather formulation:
    # output[n, oc, oh, ow] = sum_{ic, kh, kw} input[n, ic, ih, iw] * weight[ic, oc, kh, kw]
    # where ih = (oh + padding - kh) / stride if (oh + padding - kh) % stride == 0
    #       iw = (ow + padding - kw) / stride if (ow + padding - kw) % stride == 0

    # pre-compute (oh + padding) and (ow + padding)
    oh_pad = oh + PADDING  # [BLOCK_HW]
    ow_pad = ow + PADDING  # [BLOCK_HW]

    for kh in tl.static_range(KH):
        ih_num = oh_pad - kh
        ih = ih_num // STRIDE
        ih_valid = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(KW):
            iw_num = ow_pad - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW)
            valid = ih_valid & iw_valid  # [BLOCK_HW]

            # Loop over input channels in tiles
            for ic_start in range(0, IC, BLOCK_IC):
                ic_offsets = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offsets < IC

                # Load input: shape [BLOCK_HW, BLOCK_IC]
                # x[n, ic, ih, iw]
                x_offsets = (
                    pid_n * IC * IH * IW
                    + ic_offsets[None, :] * (IH * IW)
                    + ih[:, None] * IW
                    + iw[:, None]
                )
                x_mask = valid[:, None] & ic_mask[None, :] & hw_mask[:, None]
                x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

                # Load weight: shape [BLOCK_IC, BLOCK_OC]
                # weight layout: [IC, OC, KH, KW]
                w_offsets = (
                    ic_offsets[:, None] * (OC * KH * KW)
                    + oc_offsets[None, :] * (KH * KW)
                    + kh * KW + kw
                )
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Add bias
    bias = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0)
    acc = acc + bias[None, :]

    # Mish: x * tanh(softplus(x))
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(tl.where(acc > 20.0, 0.0, acc))))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = acc * th
    y = y + ADD_VALUE
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * SCALE

    # Store output: [N, OC, OH, OW]
    out_offsets = (
        pid_n * OC * OH * OW
        + oc_offsets[None, :] * (OH * OW)
        + oh[:, None] * OW
        + ow[:, None]
    )
    out_mask = hw_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offsets, y, mask=out_mask)


def conv_transpose2d_fused(x, weight, bias, stride, padding, output_padding, add_value, scale):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 64
    BLOCK_HW = 64
    BLOCK_IC = 16

    num_hw_blocks = (OH * OW + BLOCK_HW - 1) // BLOCK_HW
    num_oc_blocks = (OC + BLOCK_OC - 1) // BLOCK_OC

    grid = (num_hw_blocks, N, num_oc_blocks)

    conv_transpose2d_fused_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, padding,
        float(add_value), float(scale),
        BLOCK_OC=BLOCK_OC, BLOCK_HW=BLOCK_HW, BLOCK_IC=BLOCK_IC,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def forward(self, x):
        return conv_transpose2d_fused(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.add_value,
            self.scale,
        )