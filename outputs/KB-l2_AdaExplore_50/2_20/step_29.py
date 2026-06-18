import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_gather_kernel(
    x_ptr, w_ptr, cb_ptr, bb_ptr, out_ptr,
    N, IC, D_in, H_in, W_in,
    OC, D_out, H_out, W_out,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_sp = tl.program_id(1)
    pid_n = tl.program_id(2)

    # pid encodes oc tile
    oc_start = pid * BLOCK_OC
    offs_oc = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < OC

    sp_start = pid_sp * BLOCK_SP
    offs_sp = sp_start + tl.arange(0, BLOCK_SP)
    sp_total = D_out * H_out * W_out
    sp_mask = offs_sp < sp_total

    # decode spatial index
    d_out = offs_sp // (H_out * W_out)
    rem = offs_sp - d_out * (H_out * W_out)
    h_out = rem // W_out
    w_out = rem - h_out * W_out

    # accumulator: [BLOCK_SP, BLOCK_OC]
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # for each kernel position
    for kd in tl.static_range(0, KD):
        d_num = d_out + PAD - kd
        d_in = d_num // STRIDE
        d_valid = ((d_num - d_in * STRIDE) == 0) & (d_in >= 0) & (d_in < D_in)
        for kh in tl.static_range(0, KH):
            h_num = h_out + PAD - kh
            h_in = h_num // STRIDE
            h_valid = ((h_num - h_in * STRIDE) == 0) & (h_in >= 0) & (h_in < H_in)
            for kw in tl.static_range(0, KW):
                w_num = w_out + PAD - kw
                w_in = w_num // STRIDE
                w_valid = ((w_num - w_in * STRIDE) == 0) & (w_in >= 0) & (w_in < W_in)

                spatial_valid = d_valid & h_valid & w_valid & sp_mask  # [BLOCK_SP]

                # input offset into x[n, ic, d_in, h_in, w_in], for ic loop
                # we'll do dot product over IC
                # x_ptrs: [BLOCK_SP, IC] -- gather per spatial_valid
                # weight ptrs: w[ic, oc, kd, kh, kw], shape (IC, OC, KD, KH, KW)
                # We'll reduce over IC as a loop.

                # Compute base for x: pid_n * IC*D_in*H_in*W_in + ic*D_in*H_in*W_in + d_in*H_in*W_in + h_in*W_in + w_in
                base_spatial = d_in * (H_in * W_in) + h_in * W_in + w_in  # [BLOCK_SP]
                # safe index
                base_spatial = tl.where(spatial_valid, base_spatial, 0)
                x_base = pid_n * (IC * D_in * H_in * W_in) + base_spatial  # [BLOCK_SP]

                # weight base for (kd,kh,kw): index = ic*OC*KD*KH*KW + oc*KD*KH*KW + kd*KH*KW + kh*KW + kw
                w_kpos = kd * (KH * KW) + kh * KW + kw

                # Loop over IC
                for ic in range(0, IC):
                    x_off = x_base + ic * (D_in * H_in * W_in)  # [BLOCK_SP]
                    x_val = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)  # [BLOCK_SP]

                    w_off = ic * (OC * KD * KH * KW) + offs_oc * (KD * KH * KW) + w_kpos  # [BLOCK_OC]
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += x_val[:, None] * w_val[None, :]

    # add conv bias
    cb = tl.load(cb_ptr + offs_oc, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    bb = tl.load(bb_ptr + offs_oc, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    y = acc + cb[None, :]
    # epilogue: (2y + bb)*y + y
    out_val = (2.0 * y + bb[None, :]) * y + y

    # store: out[n, oc, d_out, h_out, w_out]
    # out shape: N, OC, D_out, H_out, W_out
    out_off = (pid_n * OC * sp_total)[None, None] if False else 0
    out_offsets = (pid_n * OC * sp_total) + offs_oc[None, :] * sp_total + offs_sp[:, None]
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offsets, out_val, mask=out_mask)


def conv_transpose3d_fused(x, weight, conv_bias, extra_bias, stride, padding, output_padding):
    N, IC, D_in, H_in, W_in = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    D_out = (D_in - 1) * stride - 2 * padding + KD + output_padding
    H_out = (H_in - 1) * stride - 2 * padding + KH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 64

    sp_total = D_out * H_out * W_out
    grid = (
        (OC + BLOCK_OC - 1) // BLOCK_OC,
        (sp_total + BLOCK_SP - 1) // BLOCK_SP,
        N,
    )

    extra_bias_flat = extra_bias.contiguous().view(-1)

    conv_transpose3d_gather_kernel[grid](
        x, weight, conv_bias, extra_bias_flat, out,
        N, IC, D_in, H_in, W_in,
        OC, D_out, H_out, W_out,
        KD, KH, KW,
        stride, padding,
        BLOCK_OC, BLOCK_SP,
        num_warps=4, num_stages=2,
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
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        return conv_transpose3d_fused(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
        )