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
    BLOCK_M: tl.constexpr,  # output spatial tile size
    BLOCK_N: tl.constexpr,  # OC tile size
    BLOCK_K: tl.constexpr,  # IC tile size
):
    # program ids
    pid_m = tl.program_id(0)  # spatial tile
    pid_n = tl.program_id(1)  # OC tile
    pid_b = tl.program_id(2)  # batch

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output spatial idx
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channel idx

    OHW = OH * OW
    mask_m = offs_m < OHW
    mask_n = offs_n < OC

    oh = offs_m // OW
    ow = offs_m % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # for each kernel position, find the input position that contributes
    # ih*STRIDE - PAD + kh = oh  =>  ih = (oh + PAD - kh) / STRIDE
    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        valid_h = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            valid_w = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW)
            valid = valid_h & valid_w & mask_m  # [BLOCK_M]

            # GEMM: sum over IC of x[b, ic, ih, iw] * w[ic, oc, kh, kw]
            # x_ptr layout: [N, IC, IH, IW] contiguous
            # w_ptr layout: [IC, OC, KH, KW] contiguous
            ih_safe = tl.where(valid, ih, 0)
            iw_safe = tl.where(valid, iw, 0)

            for ic_start in range(0, IC, BLOCK_K):
                offs_k = ic_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k < IC

                # load x [BLOCK_M, BLOCK_K]
                x_offs = (pid_b * IC * IH * IW
                          + offs_k[None, :] * IH * IW
                          + ih_safe[:, None] * IW
                          + iw_safe[:, None])
                x_mask = valid[:, None] & mask_k[None, :]
                x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

                # load w [BLOCK_K, BLOCK_N]
                w_offs = (offs_k[:, None] * OC * KH * KW
                          + offs_n[None, :] * KH * KW
                          + kh * KW + kw)
                w_mask = mask_k[:, None] & mask_n[None, :]
                w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

                acc += tl.dot(x_tile, w_tile)

    # add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # store output [N, OC, OH, OW]
    out_offs = (pid_b * OC * OHW
                + offs_n[None, :] * OHW
                + offs_m[:, None])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv_transpose2d_triton(x, weight, bias, stride, padding, output_padding, kernel_size):
    N, IC, IH, IW = x.shape
    OC = weight.shape[1]
    KH = KW = kernel_size
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(OH * OW, BLOCK_M), triton.cdiv(OC, BLOCK_N), N)

    conv_transpose2d_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, padding,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return out


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, HW,
    scaling_factor: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // HW
    hw = pid % HW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = n * C * HW + hw
    x_ptrs = x_ptr + base + offs_c * HW

    x = tl.load(x_ptrs, mask=mask_c, other=-float('inf'))
    x_f = x.to(tl.float32)

    max_val = tl.max(x_f, axis=0)
    e = tl.exp(x_f - max_val)
    e = tl.where(mask_c, e, 0.0)
    sum_e = tl.sum(e, axis=0)
    sm = e / sum_e

    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    y = (sm + b) * scaling_factor
    out = 1.0 / (1.0 + tl.exp(-y))

    out_ptrs = out_ptr + base + offs_c * HW
    tl.store(out_ptrs, out.to(x.dtype), mask=mask_c)


def fused_softmax_bias_scale_sigmoid(x, bias, scaling_factor):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)

    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * HW,)

    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias, out,
        N, C, HW,
        float(scaling_factor),
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)
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
        w = self.conv_transpose.weight.contiguous()
        cb = self.conv_transpose.bias.contiguous()
        y = conv_transpose2d_triton(
            x, w, cb,
            self.stride, self.padding, self.output_padding, self.kernel_size,
        )
        bias_flat = self.bias.view(-1).contiguous()
        return fused_softmax_bias_scale_sigmoid(y, bias_flat, self.scaling_factor)