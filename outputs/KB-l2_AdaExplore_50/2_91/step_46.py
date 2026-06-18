import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ConvTranspose2d as gather-based direct conv:
# For each output (n, oc, oh, ow):
#   y = bias[oc] + sum over (ic, kh, kw) of x[n, ic, ih, iw] * w[ic, oc, kh, kw]
# where ih*stride = oh + pad - kh and iw*stride = ow + pad - kw, divisibility required.
# We tile (N*OH*OW) on M-axis and OC on N-axis. Reduction axis is IC*KH*KW.
# Output is written in channels-last layout (N, OH, OW, OC) for fast softmax epilogue.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
    ],
    key=['N', 'OC', 'IC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv_transpose_2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    # strides for x (NCHW): n_stride_x, c_stride_x, h_stride_x, w_stride_x = IC*IH*IW, IH*IW, IW, 1
    # weight layout (IC, OC, KH, KW): w_stride_ic = OC*KH*KW, w_stride_oc=KH*KW, w_stride_kh=KW, w_stride_kw=1
    # out layout NHWC: out[n, oh, ow, oc] = out_ptr[((n*OH+oh)*OW+ow)*OC+oc]
    M, K_TOTAL,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # over N*OH*OW
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # over OC
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < OC

    # decompose offs_m -> (n, oh, ow)
    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHW = KH * KW
    K_TOTAL_VAL = IC * KHW

    for k_start in range(0, K_TOTAL_VAL, BLOCK_K):
        k_off = k_start + offs_k  # [BLOCK_K]
        k_mask = k_off < K_TOTAL_VAL

        ic = k_off // KHW
        kk = k_off % KHW
        kh = kk // KW
        kw = kk % KW

        # compute input position for each (m, k) pair
        # ih_num = oh + PAD - kh; must be divisible by STRIDE
        ih_num = oh_idx[:, None] + PAD - kh[None, :]
        iw_num = ow_idx[:, None] + PAD - kw[None, :]

        ih = ih_num // STRIDE
        iw = iw_num // STRIDE

        valid = (ih_num % STRIDE == 0) & (iw_num % STRIDE == 0)
        valid = valid & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
        valid = valid & mask_m[:, None] & k_mask[None, :]

        # x[n, ic, ih, iw]
        x_offset = n_idx[:, None] * (IC * IH * IW) + ic[None, :] * (IH * IW) + ih * IW + iw
        x_vals = tl.load(x_ptr + x_offset, mask=valid, other=0.0)  # [BLOCK_M, BLOCK_K]

        # w[ic, oc, kh, kw]: shape [BLOCK_K, BLOCK_N]
        w_offset = ic[:, None] * (OC * KHW) + offs_n[None, :] * KHW + kh[:, None] * KW + kw[:, None]
        w_mask = k_mask[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # add bias
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # store to NHWC layout: out[n, oh, ow, oc]
    out_offset = (n_idx[:, None] * (OH * OW) + oh_idx[:, None] * OW + ow_idx[:, None]) * OC + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


@triton.jit
def fused_softmax_bias_scale_sigmoid_nhwc_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, HW,
    SCALE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, hw); channels are contiguous in NHWC
    pid = tl.program_id(0)
    n = pid // HW
    hw = pid % HW

    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C

    base = (n * HW + hw) * C
    ptrs = x_ptr + base + offs_c

    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    soft = e / s

    b = tl.load(bias_ptr + offs_c, mask=mask, other=0.0)
    y = (soft + b) * SCALE
    out = 1.0 / (1.0 + tl.exp(-y))

    tl.store(out_ptr + base + offs_c, out, mask=mask)


def conv_transpose_2d_triton(x, weight, bias, stride, padding, output_padding):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # output in NHWC
    out = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

    M = N * OH * OW
    K_TOTAL = IC * KH * KW

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(OC, META['BLOCK_N']))

    conv_transpose_2d_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, padding,
        M, K_TOTAL,
    )

    return out, OH, OW


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = float(scaling_factor)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight
        conv_bias = self.conv_transpose.bias

        # output NHWC tensor
        N, IC, IH, IW = x.shape
        out_nhwc, OH, OW = conv_transpose_2d_triton(
            x, weight, conv_bias, self.stride, self.padding, self.output_padding
        )

        # fused softmax over C + bias + scale + sigmoid, NHWC layout
        OC = self.out_channels
        HW = OH * OW
        BLOCK_C = triton.next_power_of_2(OC)
        bias_flat = self.bias.contiguous().view(-1)
        result_nhwc = torch.empty_like(out_nhwc)
        grid = (N * HW,)
        fused_softmax_bias_scale_sigmoid_nhwc_kernel[grid](
            out_nhwc, bias_flat, result_nhwc,
            N, OC, HW,
            SCALE=self.scaling_factor,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        # convert back to NCHW
        result = result_nhwc.permute(0, 3, 1, 2).contiguous()
        return result