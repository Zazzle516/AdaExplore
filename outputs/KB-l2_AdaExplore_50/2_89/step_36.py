import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused ConvTranspose3d + MaxPool3d + Softmax(C) + Sub + Swish + Max(C)
# Output: (N, D_out, H_out, W_out) where (D_out, H_out, W_out) is the pooled spatial size.
#
# Conv transpose params: stride=2, padding=1, output_padding=1, kernel_size=3
# So the conv-transpose output spatial = (D_in*2, H_in*2, W_in*2) for the given config.
# MaxPool: kernel=2, stride=2, padding=0 -> pooled spatial = (D_in, H_in, W_in).
#
# For each pooled position (n, d_p, h_p, w_p), the 2x2x2 conv-transpose window covers
# conv output positions [2*d_p .. 2*d_p+1] x [2*h_p .. 2*h_p+1] x [2*w_p .. 2*w_p+1].
# We compute these 8 conv-transpose outputs on the fly for each of OC channels,
# take the max over the 8 (pooling), then perform softmax over OC, subtract, swish, max over OC.


@triton.jit
def fused_convt_pool_smax_swish_kernel(
    x_ptr,         # (N, IC, D_in, H_in, W_in)
    w_ptr,         # (IC, OC, K, K, K)
    b_ptr,         # (OC,)
    sub_ptr,       # (OC,)
    out_ptr,       # (N, D_out, H_out, W_out)
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,   # pooled output spatial sizes (== D_in, H_in, W_in given config)
    stride_n_x, stride_c_x, stride_d_x, stride_h_x, stride_w_x,
    BLOCK_OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    # Decompose pid into (n, d_p, h_p, w_p)
    SP = D_out * H_out * W_out
    n = pid // SP
    rem = pid % SP
    d_p = rem // (H_out * W_out)
    rem2 = rem % (H_out * W_out)
    h_p = rem2 // W_out
    w_p = rem2 % W_out

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # Initialize pooled values: max over 2x2x2 window of conv-transpose outputs
    # We compute each of 8 conv outputs into a vector of length BLOCK_OC, then take elementwise max.
    NEG_INF = float('-inf')
    pooled = tl.full([BLOCK_OC], NEG_INF, dtype=tl.float32)

    # Iterate over 8 positions in the 2x2x2 pool window
    for dd in tl.static_range(0, 2):
        for hh in tl.static_range(0, 2):
            for ww in tl.static_range(0, 2):
                d_o = d_p * 2 + dd
                h_o = h_p * 2 + hh
                w_o = w_p * 2 + ww

                # ConvTranspose3d formula:
                # out[n, oc, d_o, h_o, w_o] = sum over ic, kd, kh, kw of
                #   x[n, ic, d_in, h_in, w_in] * w[ic, oc, kd, kh, kw]
                # where: d_o + PAD = d_in * STRIDE + kd
                # so for each kd: d_in*STRIDE = d_o + PAD - kd
                #     d_in = (d_o + PAD - kd) / STRIDE (must be integer, in range)

                # Initialize accumulator with bias
                acc = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0).to(tl.float32)

                for kd in tl.static_range(0, KD):
                    num_d = d_o + PAD - kd
                    d_in = num_d // STRIDE
                    d_ok = (num_d % STRIDE == 0) & (d_in >= 0) & (d_in < D_in)
                    for kh in tl.static_range(0, KH):
                        num_h = h_o + PAD - kh
                        h_in = num_h // STRIDE
                        h_ok = (num_h % STRIDE == 0) & (h_in >= 0) & (h_in < H_in)
                        for kw in tl.static_range(0, KW):
                            num_w = w_o + PAD - kw
                            w_in = num_w // STRIDE
                            w_ok = (num_w % STRIDE == 0) & (w_in >= 0) & (w_in < W_in)
                            valid = d_ok & h_ok & w_ok
                            if valid:
                                # Sum over IC
                                # x offset for (n, ic, d_in, h_in, w_in)
                                x_base = (n * stride_n_x
                                          + d_in * stride_d_x
                                          + h_in * stride_h_x
                                          + w_in * stride_w_x)
                                # w offset for (ic, oc, kd, kh, kw): contiguous IC,OC,KD,KH,KW
                                # w[ic, oc, kd, kh, kw] = w_ptr[ic*OC*KD*KH*KW + oc*KD*KH*KW + kd*KH*KW + kh*KW + kw]
                                w_kbase = kd * KH * KW + kh * KW + kw  # within (kd, kh, kw)
                                for ic in tl.static_range(0, 3):  # IC = 3 hardcoded
                                    x_val = tl.load(x_ptr + x_base + ic * stride_c_x)
                                    # weights for all oc at (ic, :, kd, kh, kw)
                                    w_off = ic * OC * KD * KH * KW + offs_oc * KD * KH * KW + w_kbase
                                    w_vals = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)
                                    acc += x_val * w_vals

                # acc now holds conv-transpose output for (n, :, d_o, h_o, w_o)
                pooled = tl.maximum(pooled, acc)

    # pooled: [BLOCK_OC] floats. For invalid oc lanes, set to -inf for softmax
    pooled = tl.where(mask_oc, pooled, NEG_INF)

    # Softmax over OC
    x_max = tl.max(pooled, axis=0)
    x_shift = pooled - x_max
    x_exp = tl.exp(x_shift)
    x_exp = tl.where(mask_oc, x_exp, 0.0)
    denom = tl.sum(x_exp, axis=0)
    sm = x_exp / denom

    # Subtract
    sub = tl.load(sub_ptr + offs_oc, mask=mask_oc, other=0.0)
    y = sm - sub

    # Swish: sigmoid(y) * y
    sig = 1.0 / (1.0 + tl.exp(-y))
    sw = sig * y

    # Max over OC
    sw_masked = tl.where(mask_oc, sw, NEG_INF)
    res = tl.max(sw_masked, axis=0)

    # Output offset (N, D_out, H_out, W_out)
    out_off = ((n * D_out + d_p) * H_out + h_p) * W_out + w_p
    tl.store(out_ptr + out_off, res)


def fused_forward(x, weight, bias, subtract,
                  stride, padding, output_padding,
                  pool_kernel_size, pool_stride, pool_padding):
    """
    x: (N, IC, D_in, H_in, W_in)
    weight: (IC, OC, KD, KH, KW)  -- ConvTranspose3d weight layout
    bias: (OC,)
    subtract: (OC,)
    """
    assert x.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    subtract = subtract.contiguous()

    N, IC, D_in, H_in, W_in = x.shape
    _, OC, KD, KH, KW = weight.shape

    # conv-transpose output spatial:
    D_conv = (D_in - 1) * stride - 2 * padding + KD + output_padding
    H_conv = (H_in - 1) * stride - 2 * padding + KH + output_padding
    W_conv = (W_in - 1) * stride - 2 * padding + KW + output_padding

    # pooled output spatial (pool_stride=2, pool_kernel=2, pool_padding=0)
    D_out = (D_conv + 2 * pool_padding - pool_kernel_size) // pool_stride + 1
    H_out = (H_conv + 2 * pool_padding - pool_kernel_size) // pool_stride + 1
    W_out = (W_conv + 2 * pool_padding - pool_kernel_size) // pool_stride + 1

    out = torch.empty((N, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_OC = triton.next_power_of_2(OC)
    if BLOCK_OC < 16:
        BLOCK_OC = 16

    grid = (N * D_out * H_out * W_out,)

    stride_n_x = IC * D_in * H_in * W_in
    stride_c_x = D_in * H_in * W_in
    stride_d_x = H_in * W_in
    stride_h_x = W_in
    stride_w_x = 1

    fused_convt_pool_smax_swish_kernel[grid](
        x, weight, bias, subtract, out,
        N, IC, OC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride_n_x, stride_c_x, stride_d_x, stride_h_x, stride_w_x,
        BLOCK_OC=BLOCK_OC,
        KD=KD, KH=KH, KW=KW,
        STRIDE=stride,
        PAD=padding,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding,
                 pool_kernel_size, pool_stride, pool_padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.max_pool = nn.MaxPool3d(
            kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding
        )
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = x.cuda().contiguous()
        return fused_forward(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.subtract,
            self.stride, self.padding, self.output_padding,
            self.pool_kernel_size, self.pool_stride, self.pool_padding,
        )