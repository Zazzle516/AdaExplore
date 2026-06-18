import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 32}, num_warps=8, num_stages=2),
    ],
    key=['C', 'HW'],
)
@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, HW,
    scaling_factor: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    num_hw_blocks = (HW + BLOCK_HW - 1) // BLOCK_HW
    n = pid // num_hw_blocks
    hw_blk = pid % num_hw_blocks
    hw_start = hw_blk * BLOCK_HW

    offs_c = tl.arange(0, BLOCK_C)
    offs_hw = hw_start + tl.arange(0, BLOCK_HW)

    mask_c = offs_c < C
    mask_hw = offs_hw < HW
    mask = mask_c[:, None] & mask_hw[None, :]

    base = n * C * HW
    x_ptrs = x_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]

    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)

    m = tl.max(x, axis=0)
    x_shift = x - m[None, :]
    e = tl.exp(x_shift)
    s = tl.sum(e, axis=0)
    sm = e / s[None, :]

    y = (sm + b[:, None]) * scaling_factor
    y = 1.0 / (1.0 + tl.exp(-y))

    tl.store(out_ptr + base + offs_c[:, None] * HW + offs_hw[None, :], y, mask=mask)


def fused_post_conv(x, bias, scaling_factor):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    BLOCK_C = triton.next_power_of_2(C)
    grid = lambda meta: (N * triton.cdiv(HW, meta['BLOCK_HW']),)
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias.view(-1), out,
        N, C, HW,
        scaling_factor=float(scaling_factor),
        BLOCK_C=BLOCK_C,
    )
    return out


# Transposed conv as scatter-add. For stride=2, padding=1, output_padding=1, kernel=4:
# output H_out = (H_in - 1) * 2 - 2*1 + 4 + 1 = 2*H_in + 1... wait let me recompute
# H_out = (H_in - 1)*stride - 2*padding + kernel_size + output_padding
#       = (64 - 1)*2 - 2 + 4 + 1 = 126 - 2 + 4 + 1 = 129
# Hmm, but we should compute generically.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1, 'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 1, 'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 1, 'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1, 'BLOCK_OC': 128, 'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 1, 'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 1, 'BLOCK_OC': 64, 'BLOCK_HW': 32}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'OC', 'H_out', 'W_out'],
)
@triton.jit
def conv_transpose2d_gather_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H_in, W_in,
    OC, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_oc = tl.program_id(2)

    HW_out = H_out * W_out
    hw_start = pid * BLOCK_HW
    offs_hw = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW_out
    oh = offs_hw // W_out
    ow = offs_hw % W_out

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # Load bias [BLOCK_OC]
    bias_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = bias_vals[:, None] + tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # For each kernel position (kh, kw), compute input position:
    # i_h = (oh + PAD_H - kh) / STRIDE_H  if exact division
    # i_w = (ow + PAD_W - kw) / STRIDE_W  if exact division
    # Then for each ic in tile, accumulate x[n, ic, i_h, i_w] * w[ic, oc, kh, kw]
    
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_num = oh + PAD_H - kh
            iw_num = ow + PAD_W - kw
            ih = ih_num // STRIDE_H
            iw = iw_num // STRIDE_W
            valid = (ih_num % STRIDE_H == 0) & (iw_num % STRIDE_W == 0) & \
                    (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in) & mask_hw

            # Loop over ic in tiles of BLOCK_IC
            for ic_start in range(0, IC, BLOCK_IC):
                offs_ic = ic_start + tl.arange(0, BLOCK_IC)
                mask_ic = offs_ic < IC

                # Load x: shape [BLOCK_IC, BLOCK_HW]
                # x[n, ic, ih, iw] - ih,iw vary per HW position
                x_offs = pid_n * IC * H_in * W_in + offs_ic[:, None] * (H_in * W_in) + ih[None, :] * W_in + iw[None, :]
                x_mask = mask_ic[:, None] & valid[None, :]
                x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

                # Load weight: w[ic, oc, kh, kw], shape [BLOCK_IC, BLOCK_OC]
                w_offs = offs_ic[:, None] * (OC * KH * KW) + offs_oc[None, :] * (KH * KW) + kh * KW + kw
                w_mask = mask_ic[:, None] & mask_oc[None, :]
                w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

                # GEMM: acc[oc, hw] += w[ic, oc].T @ x[ic, hw]
                acc += tl.dot(tl.trans(w_vals), x_vals)

    # Store output [BLOCK_OC, BLOCK_HW]
    out_offs = pid_n * OC * HW_out + offs_oc[:, None] * HW_out + offs_hw[None, :]
    out_mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv_transpose2d_triton(x, weight, bias, stride, padding, output_padding):
    N, IC, H_in, W_in = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    H_out = (H_in - 1) * stride - 2 * padding + KH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_IC = 16
    if IC % 32 == 0:
        BLOCK_IC = 32

    grid = lambda meta: (
        triton.cdiv(H_out * W_out, meta['BLOCK_HW']),
        N,
        triton.cdiv(OC, meta['BLOCK_OC']),
    )

    conv_transpose2d_gather_kernel[grid](
        x, weight, bias, out,
        N, IC, H_in, W_in,
        OC, H_out, W_out,
        KH=KH, KW=KW,
        STRIDE_H=stride, STRIDE_W=stride,
        PAD_H=padding, PAD_W=padding,
        BLOCK_IC=BLOCK_IC,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous()
        # Use custom transposed conv
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