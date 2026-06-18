import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Input parameters:
# N=128, IC=64, IH=IW=64
# OC=128, K=4, stride=2, padding=1, output_padding=1
# Output: OH = (IH-1)*stride - 2*pad + K + output_padding = 63*2 - 2 + 4 + 1 = 129
# OW = 129
# So output shape: (128, 128, 129, 129)


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,        # [N, IC, IH, IW]
    w_ptr,        # [IC, OC, K, K]
    b_ptr,        # [OC]
    out_ptr,      # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # Grid: (N, OC // BLOCK_OC, (OH*OW) // BLOCK_SP)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    offs_sp = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oh = offs_sp // OW   # [BLOCK_SP]
    ow = offs_sp % OW

    mask_oc = offs_oc < OC
    mask_sp = offs_sp < OH * OW

    # accumulator
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For each (oh, ow), input position contributions come from:
    # ih*stride - pad + kh = oh -> ih = (oh + pad - kh) / stride; valid if divisible.
    # Iterate over kh, kw, ic.
    # ih_num = oh + pad - kh; if ih_num % stride == 0 and 0 <= ih_num/stride < IH

    # We will iterate over kh, kw
    for kh in tl.static_range(0, K):
        ih_num = oh + PAD - kh  # [BLOCK_SP]
        ih = ih_num // STRIDE
        ih_valid = (ih_num >= 0) & (ih_num % STRIDE == 0) & (ih < IH)

        for kw in tl.static_range(0, K):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num >= 0) & (iw_num % STRIDE == 0) & (iw < IW)
            valid = ih_valid & iw_valid  # [BLOCK_SP]

            # safe indices
            ih_safe = tl.where(ih_valid, ih, 0)
            iw_safe = tl.where(iw_valid, iw, 0)

            # Load weight block [IC, OC] for this (kh, kw):
            # weight layout: [IC, OC, K, K]; offset: ic*OC*K*K + oc*K*K + kh*K + kw
            # We loop over IC in tiles of BLOCK_IC
            for ic_start in range(0, IC, BLOCK_IC):
                offs_ic = ic_start + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
                mask_ic = offs_ic < IC

                # Load x[N, ic, ih, iw] for each sp: shape [BLOCK_SP, BLOCK_IC]
                x_offs = (
                    pid_n * IC * IH * IW
                    + offs_ic[None, :] * IH * IW
                    + ih_safe[:, None] * IW
                    + iw_safe[:, None]
                )
                x_mask = valid[:, None] & mask_ic[None, :]
                x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)  # [BLOCK_SP, BLOCK_IC]

                # Load weight[ic, oc, kh, kw]: shape [BLOCK_IC, BLOCK_OC]
                w_offs = (
                    offs_ic[:, None] * OC * K * K
                    + offs_oc[None, :] * K * K
                    + kh * K
                    + kw
                )
                w_mask = mask_ic[:, None] & mask_oc[None, :]
                w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]

                acc += tl.dot(x_vals, w_vals)

    # Add bias
    b = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)  # [BLOCK_OC]
    acc += b[None, :]

    # Store: out[N, OC, OH, OW]
    out_offs = (
        pid_n * OC * OH * OW
        + offs_oc[None, :] * OH * OW
        + offs_sp[:, None]
    )
    out_mask = mask_sp[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    N, C, H, W,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    HW = H * W
    n = pid // HW
    hw = pid % HW

    base = n * C * HW + hw

    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C

    x_ptrs = x_ptr + base + offs_c * HW
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    max_val = tl.max(x, axis=0)
    x_shift = x - max_val
    exp_x = tl.exp(x_shift)
    exp_x = tl.where(mask, exp_x, 0.0)
    sum_val = tl.sum(exp_x, axis=0)
    inv_sum = 1.0 / sum_val
    sm = exp_x * inv_sum

    bias = tl.load(bias_ptr + offs_c, mask=mask, other=0.0)

    y = (sm + bias) * scaling_factor
    y = 1.0 / (1.0 + tl.exp(-y))

    out_ptrs = out_ptr + base + offs_c * HW
    tl.store(out_ptrs, y, mask=mask)


def conv_transpose2d_triton(x, weight, bias_conv, stride, padding, output_padding):
    N, IC, IH, IW = x.shape
    IC_w, OC, K, K2 = weight.shape
    assert IC == IC_w
    assert K == K2

    OH = (IH - 1) * stride - 2 * padding + K + output_padding
    OW = (IW - 1) * stride - 2 * padding + K + output_padding

    x = x.contiguous()
    weight = weight.contiguous()
    bias_conv = bias_conv.contiguous()

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 64
    BLOCK_SP = 64
    BLOCK_IC = 32

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

    conv_transpose2d_kernel[grid](
        x, weight, bias_conv, out,
        N, IC, IH, IW,
        OC, OH, OW,
        K=K,
        STRIDE=stride,
        PAD=padding,
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        BLOCK_IC=BLOCK_IC,
        num_warps=4,
        num_stages=2,
    )
    return out


def fused_post_conv(x, bias, scaling_factor):
    N, C, H, W = x.shape
    x = x.contiguous()
    bias_flat = bias.contiguous().view(-1)
    out = torch.empty_like(x)

    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = (N * H * W,)
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias_flat, out,
        N, C, H, W,
        float(scaling_factor),
        BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def forward(self, x):
        x = conv_transpose2d_triton(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding,
        )
        x = fused_post_conv(x, self.bias, self.scaling_factor)
        return x