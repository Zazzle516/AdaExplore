import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def convtranspose2d_gather_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H_in, W_in,
    OC, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    ADD_VALUE: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (H_out * W_out)

    h_out = sp_offs // W_out
    w_out = sp_offs % W_out

    # accumulator [BLOCK_SP, BLOCK_OC]
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For ConvTranspose2d: out[n,oc,h,w] = sum_{ic,kh,kw} x[n,ic, (h+pad-kh)/stride, (w+pad-kw)/stride] * w[ic,oc,kh,kw]
    # only when (h+pad-kh) % stride == 0 and within bounds.

    for kh in tl.static_range(0, KH):
        h_num = h_out + PAD - kh  # [BLOCK_SP]
        h_in = h_num // STRIDE
        h_valid = (h_num >= 0) & ((h_num % STRIDE) == 0) & (h_in < H_in)
        for kw in tl.static_range(0, KW):
            w_num = w_out + PAD - kw
            w_in = w_num // STRIDE
            w_valid = (w_num >= 0) & ((w_num % STRIDE) == 0) & (w_in < W_in)
            valid = h_valid & w_valid & sp_mask  # [BLOCK_SP]

            # base offset for input x[n, :, h_in, w_in]: stride IC*H_in*W_in for n, H_in*W_in for ic dim
            x_base = pid_n * IC * H_in * W_in + h_in * W_in + w_in  # [BLOCK_SP]
            # weight base for w[:, oc, kh, kw]: stride OC*KH*KW per ic
            w_base = oc_offs * KH * KW + kh * KW + kw  # [BLOCK_OC]

            for ic in range(0, IC):
                x_off = x_base + ic * H_in * W_in
                x_vals = tl.load(x_ptr + x_off, mask=valid, other=0.0)  # [BLOCK_SP]
                w_off = ic * OC * KH * KW + w_base
                w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                acc += x_vals[:, None] * w_vals[None, :]

    # add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + bias[None, :]

    # Mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = acc * th
    y = y + ADD_VALUE
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * SCALE

    # store: out[n, oc, h_out, w_out]
    out_offs = (pid_n * OC * H_out * W_out
                + oc_offs[None, :] * H_out * W_out
                + sp_offs[:, None])
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offs, y, mask=store_mask)


def convtranspose2d_fused(x, weight, bias, stride, padding, output_padding, add_value, scale):
    N, IC, H_in, W_in = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    H_out = (H_in - 1) * stride - 2 * padding + KH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 128

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(H_out * W_out, BLOCK_SP))

    convtranspose2d_gather_kernel[grid](
        x, weight, bias, out,
        N, IC, H_in, W_in,
        OC, H_out, W_out,
        KH=KH, KW=KW,
        STRIDE=stride, PAD=padding,
        ADD_VALUE=float(add_value), SCALE=float(scale),
        BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
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

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()
        return convtranspose2d_fused(
            x, w, b,
            self.stride, self.padding, self.output_padding,
            self.add_value, self.scale,
        )