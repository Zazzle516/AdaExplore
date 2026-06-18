import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    x_ptr,        # (N, IC, H_in, W_in)
    w_ptr,        # (IC, OC, KH, KW)
    b_ptr,        # (OC,)
    out_ptr,      # (N, H_out, W_out, OC)  - channels last
    N, IC, OC,
    H_in, W_in, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)
    pid_oc = tl.program_id(2)

    HW_out = H_out * W_out
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < HW_out
    h_out = sp_offs // W_out
    w_out = sp_offs % W_out

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For transposed conv: out[h,w] = sum over ic, kh, kw of
    #   x[ic, (h + PAD_H - kh)/STRIDE_H, (w + PAD_W - kw)/STRIDE_W] * w[ic, oc, kh, kw]
    # valid when (h + PAD_H - kh) is divisible by stride and in range.

    for kh in tl.static_range(KH):
        h_num = h_out + PAD_H - kh
        h_in = h_num // STRIDE_H
        h_valid = (h_num >= 0) & (h_num - h_in * STRIDE_H == 0) & (h_in >= 0) & (h_in < H_in)
        for kw in tl.static_range(KW):
            w_num = w_out + PAD_W - kw
            w_in = w_num // STRIDE_W
            w_valid = (w_num >= 0) & (w_num - w_in * STRIDE_W == 0) & (w_in >= 0) & (w_in < W_in)
            valid = h_valid & w_valid & sp_mask  # (BLOCK_SP,)

            # input offset for each spatial position: n*IC*H_in*W_in + ic*H_in*W_in + h_in*W_in + w_in
            # we'll iterate ic in blocks
            sp_in_offset = h_in * W_in + w_in  # (BLOCK_SP,)
            # weight slice: w[ic, oc_block, kh, kw]
            # walk over IC
            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                # Load x: shape (BLOCK_SP, BLOCK_IC)
                x_ptrs = x_ptr + pid_n * IC * H_in * W_in + ic_offs[None, :] * (H_in * W_in) + sp_in_offset[:, None]
                x_mask = valid[:, None] & ic_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                # Load weight: shape (BLOCK_IC, BLOCK_OC) = w[ic, oc, kh, kw]
                w_ptrs = w_ptr + ic_offs[:, None] * (OC * KH * KW) + oc_offs[None, :] * (KH * KW) + kh * KW + kw
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # add bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_vals[None, :]

    # Store: output is (N, H_out, W_out, OC), channels-last layout
    out_ptrs = (
        out_ptr
        + pid_n * (HW_out * OC)
        + sp_offs[:, None] * OC
        + oc_offs[None, :]
    )
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr,         # input: (N, H, W, C) channels-last
    bias_ptr,      # bias: (C,)
    out_ptr,       # output: (N, C, H, W) standard layout
    N, C, H, W,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    HW = H * W
    nhw = N * HW

    n = pid // HW
    hw = pid % HW
    h = hw // W
    w = hw % W

    # input base: channels-last, contiguous along C
    in_base = n * HW * C + hw * C
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    x = tl.load(x_ptr + in_base + offs_c, mask=mask_c, other=-float('inf'))

    m = tl.max(x, axis=0)
    x_shift = x - m
    e = tl.exp(x_shift)
    e = tl.where(mask_c, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    y = (sm + b) * scaling_factor
    y = 1.0 / (1.0 + tl.exp(-y))

    # store in (N, C, H, W) layout
    out_base = n * C * HW + h * W + w
    stride_c = HW
    out_ptrs = out_ptr + out_base + offs_c * stride_c
    tl.store(out_ptrs, y, mask=mask_c)


def conv_transpose_triton(x, weight, conv_bias, stride, padding, kernel_size):
    N, IC, H_in, W_in = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w
    H_out = (H_in - 1) * stride - 2 * padding + KH + (stride - 1)  # output_padding = stride - 1 here for our case
    # Actually compute properly:
    # H_out = (H_in - 1)*stride - 2*pad + kernel + output_padding
    # We'll pass H_out, W_out from outside

    raise NotImplementedError


def fused_conv_transpose_full(x, weight, conv_bias, bias, scaling_factor,
                              stride, padding, output_padding, kernel_size):
    N, IC, H_in, W_in = x.shape
    IC_w, OC, KH, KW = weight.shape
    H_out = (H_in - 1) * stride - 2 * padding + KH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + KW + output_padding

    x = x.contiguous()
    weight = weight.contiguous()
    conv_bias = conv_bias.contiguous()

    # Output of conv_transpose in channels-last layout (N, H_out, W_out, OC)
    conv_out = torch.empty((N, H_out, W_out, OC), device=x.device, dtype=x.dtype)

    HW_out = H_out * W_out
    grid_conv = lambda META: (N, triton.cdiv(HW_out, META['BLOCK_SP']), triton.cdiv(OC, META['BLOCK_OC']))

    conv_transpose_kernel[grid_conv](
        x, weight, conv_bias, conv_out,
        N, IC, OC,
        H_in, W_in, H_out, W_out,
        KH=KH, KW=KW,
        STRIDE_H=stride, STRIDE_W=stride,
        PAD_H=padding, PAD_W=padding,
    )

    # Now apply fused softmax+bias+scale+sigmoid
    out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)
    bias_flat = bias.contiguous().view(-1)

    BLOCK_C = 1
    while BLOCK_C < OC:
        BLOCK_C *= 2

    grid_sm = (N * HW_out,)
    fused_softmax_bias_scale_sigmoid_kernel[grid_sm](
        conv_out, bias_flat, out,
        N, OC, H_out, W_out,
        float(scaling_factor),
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        return fused_conv_transpose_full(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.bias,
            self.scaling_factor,
            self.stride, self.padding, self.output_padding, self.kernel_size,
        )