import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_scatter_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC, IH, IW, OH, OW, KH, KW,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # Each program: one (n, oh, ow) output position, computes all OC channels
    pid = tl.program_id(0)
    ow = pid % OW
    pid1 = pid // OW
    oh = pid1 % OH
    n = pid1 // OH

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    acc = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)

    # For each (kh, kw), find ih, iw such that ih*stride - pad + kh = oh
    # => ih = (oh + pad - kh) / stride if divisible
    for kh in tl.static_range(0, KH):
        ih_num = oh + pad_h - kh
        ih = ih_num // stride_h
        ih_valid = (ih_num % stride_h == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow + pad_w - kw
            iw = iw_num // stride_w
            iw_valid = (iw_num % stride_w == 0) & (iw >= 0) & (iw < IW)
            valid = ih_valid & iw_valid

            # Loop over IC in blocks
            for ic_start in range(0, IC, BLOCK_IC):
                offs_ic = ic_start + tl.arange(0, BLOCK_IC)
                mask_ic = offs_ic < IC

                # Load x[n, ic, ih, iw] for ic in block (vector of BLOCK_IC)
                x_offs = n * IC * IH * IW + offs_ic * IH * IW + ih * IW + iw
                x_vals = tl.load(x_ptr + x_offs, mask=mask_ic & valid, other=0.0)

                # Load w[ic, oc, kh, kw] for ic in block, oc in OC block
                # Weight shape (IC, OC, KH, KW)
                w_offs = (offs_ic[:, None] * OC * KH * KW +
                          offs_oc[None, :] * KH * KW +
                          kh * KW + kw)
                w_vals = tl.load(w_ptr + w_offs,
                                 mask=mask_ic[:, None] & mask_oc[None, :],
                                 other=0.0)
                # Contribution: sum over ic of x_vals[ic] * w_vals[ic, oc]
                acc += tl.sum(x_vals[:, None] * w_vals, axis=0)

    # acc has shape (BLOCK_OC,) = full OC vector for this spatial position
    # Now fuse: softmax over OC, +bias_post, *scale, sigmoid
    # But bias_post comes from another buffer — pass it separately
    # Store raw acc for now; epilogue done separately? Actually let's fuse here.
    tl.store(out_ptr + (n * OC * OH * OW + offs_oc * OH * OW + oh * OW + ow),
             acc, mask=mask_oc)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=3),
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
    grid = lambda META: (N * triton.cdiv(HW, META['BLOCK_HW']),)
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias.view(-1), out,
        N, C, HW,
        scaling_factor=float(scaling_factor),
        BLOCK_C=BLOCK_C,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_post_conv(x, self.bias, self.scaling_factor)
        return x