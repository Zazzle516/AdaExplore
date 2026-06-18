import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, bias_conv_ptr, bias_eps_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_D: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # program ids: (n, oc_block, spatial_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offsets = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offsets = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offsets < OC

    # Decompose spatial offsets into (od, oh, ow)
    OHW = OH * OW
    od = sp_offsets // OHW
    rem = sp_offsets % OHW
    oh = rem // OW
    ow = rem % OW
    sp_mask = sp_offsets < (OD * OHW)

    # Accumulator
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For ConvTranspose3d:
    # out[n, oc, od, oh, ow] = sum over (ic, kd, kh, kw) of
    #   x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where id*stride - pad + kd = od  =>  id = (od + pad - kd) / stride
    # only valid when (od + pad - kd) % stride == 0 and 0 <= id < ID

    # Loop over kernel positions
    for kd in tl.static_range(0, KD):
        d_num = od + PAD_D - kd
        d_valid = (d_num % STRIDE_D) == 0
        id_ = d_num // STRIDE_D
        d_in = (id_ >= 0) & (id_ < ID) & d_valid

        for kh in tl.static_range(0, KH):
            h_num = oh + PAD_H - kh
            h_valid = (h_num % STRIDE_H) == 0
            ih_ = h_num // STRIDE_H
            h_in = (ih_ >= 0) & (ih_ < IH) & h_valid

            for kw in tl.static_range(0, KW):
                w_num = ow + PAD_W - kw
                w_valid = (w_num % STRIDE_W) == 0
                iw_ = w_num // STRIDE_W
                w_in = (iw_ >= 0) & (iw_ < IW) & w_valid

                spatial_in = d_in & h_in & w_in & sp_mask  # [BLOCK_SP]

                # input base offset for [n, :, id_, ih_, iw_]
                # x layout: [N, IC, ID, IH, IW]
                in_spatial = id_ * (IH * IW) + ih_ * IW + iw_  # [BLOCK_SP]
                x_base = pid_n * (IC * ID * IH * IW) + in_spatial  # [BLOCK_SP]

                # weight offset for [:, oc, kd, kh, kw]
                # w layout: [IC, OC, KD, KH, KW]
                w_kernel_off = kd * (KH * KW) + kh * KW + kw  # scalar
                w_base = oc_offsets * (KD * KH * KW) + w_kernel_off  # [BLOCK_OC]

                # Loop over IC, accumulating outer product
                for ic in range(0, IC):
                    x_ptrs = x_ptr + x_base + ic * (ID * IH * IW)
                    x_vals = tl.load(x_ptrs, mask=spatial_in, other=0.0)  # [BLOCK_SP]

                    w_ptrs = w_ptr + w_base + ic * (OC * KD * KH * KW)
                    w_vals = tl.load(w_ptrs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += x_vals[:, None] * w_vals[None, :]

    # Add conv bias
    bconv = tl.load(bias_conv_ptr + oc_offsets, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    y = acc + bconv[None, :]

    # Epilogue: out = (2*y + bias_eps) * y + y
    beps = tl.load(bias_eps_ptr + oc_offsets, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    result = (2.0 * y + beps[None, :]) * y + y

    # Store: out layout [N, OC, OD, OH, OW]
    out_spatial = od * OHW + oh * OW + ow  # [BLOCK_SP]
    out_base = pid_n * (OC * OD * OHW)  # scalar

    out_ptrs = out_ptr + out_base + oc_offsets[None, :] * (OD * OHW) + out_spatial[:, None]
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptrs, result, mask=store_mask)


def conv_transpose3d_fused(x, weight, bias_conv, bias_eps,
                            stride, padding, output_padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape

    OD = (ID - 1) * stride[0] - 2 * padding[0] + KD + output_padding[0]
    OH = (IH - 1) * stride[1] - 2 * padding[1] + KH + output_padding[1]
    OW = (IW - 1) * stride[2] - 2 * padding[2] + KW + output_padding[2]

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 128

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OD * OH * OW, BLOCK_SP))

    conv_transpose3d_kernel[grid](
        x, weight, bias_conv, bias_eps, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = tuple(stride)
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = tuple(padding)
        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding, output_padding)
        else:
            self.output_padding = tuple(output_padding)

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        bias_conv = self.conv_transpose.bias.contiguous()
        bias_eps = self.bias.contiguous().view(-1)

        return conv_transpose3d_fused(
            x, weight, bias_conv, bias_eps,
            self.stride, self.padding, self.output_padding,
        )