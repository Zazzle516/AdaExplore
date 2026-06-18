import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'OH', 'OW', 'IC'],
)
@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    ic_range = tl.arange(0, BLOCK_IC)

    oh = sp_offs // OW
    ow = sp_offs % OW

    mask_oc = oc_offs < OC
    mask_sp = sp_offs < OH * OW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # x: [N, IC, IH, IW], w: [IC, OC, KH, KW]
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_num = oh + PAD - kh
            iw_num = ow + PAD - kw
            ih = ih_num // STRIDE
            iw = iw_num // STRIDE
            valid_sp = (ih_num % STRIDE == 0) & (iw_num % STRIDE == 0) & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & mask_sp

            # x_idx for [BLOCK_SP] -- needs broadcast with ic
            sp_idx = ih * IW + iw  # [BLOCK_SP]
            # w_idx for [BLOCK_OC] -- needs broadcast with ic
            w_oc_idx = oc_offs * KH * KW + kh * KW + kw  # [BLOCK_OC]

            for ic_block in range(0, tl.cdiv(IC, BLOCK_IC)):
                ic_offs = ic_block * BLOCK_IC + ic_range  # [BLOCK_IC]
                mask_ic = ic_offs < IC

                # x[n, ic, ih, iw] -> [BLOCK_IC, BLOCK_SP]
                x_ptrs = x_ptr + pid_n * IC * IH * IW + ic_offs[:, None] * IH * IW + sp_idx[None, :]
                x_mask = mask_ic[:, None] & valid_sp[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_IC, BLOCK_SP]

                # w[ic, oc, kh, kw] -> [BLOCK_OC, BLOCK_IC]
                w_ptrs = w_ptr + ic_offs[None, :] * OC * KH * KW + w_oc_idx[:, None]
                w_mask = mask_oc[:, None] & mask_ic[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_OC, BLOCK_IC]

                acc += tl.dot(w_vals, x_vals, allow_tf32=True)

    b = tl.load(b_ptr + oc_offs, mask=mask_oc, other=0.0)
    acc += b[:, None]

    out_idx = pid_n * OC * OH * OW + oc_offs[:, None] * (OH * OW) + sp_offs[None, :]
    out_mask = mask_oc[:, None] & mask_sp[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def conv_transpose_triton(x, weight, bias, stride, padding, OH, OW):
    N, IC, IH, IW = x.shape
    _, OC, KH, KW = weight.shape
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(OH * OW, META['BLOCK_SP']))
    conv_transpose_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, padding,
    )
    return out


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, HW,
    scaling_factor,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    blocks_per_n = tl.cdiv(HW, BLOCK_HW)
    n = pid_nhw // blocks_per_n
    hw_block = pid_nhw % blocks_per_n
    hw_start = hw_block * BLOCK_HW

    offs_hw = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = n * C * HW
    x_ptrs = x_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]
    mask_2d = mask_c[:, None] & mask_hw[None, :]

    x = tl.load(x_ptrs, mask=mask_2d, other=-float('inf'))

    max_val = tl.max(x, axis=0)
    x_shift = x - max_val[None, :]
    exp_x = tl.exp(x_shift)
    exp_x = tl.where(mask_2d, exp_x, 0.0)
    sum_val = tl.sum(exp_x, axis=0)
    sm = exp_x / sum_val[None, :]

    bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    y = (sm + bias[:, None]) * scaling_factor
    y = 1.0 / (1.0 + tl.exp(-y))

    out_ptrs = out_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]
    tl.store(out_ptrs, y, mask=mask_2d)


def fused_post_conv(x, bias, scaling_factor):
    N, C, H, W = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    bias_flat = bias.contiguous().view(-1)
    out = torch.empty_like(x)
    HW = H * W

    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    BLOCK_HW = 256
    blocks_per_n = (HW + BLOCK_HW - 1) // BLOCK_HW
    grid = (N * blocks_per_n,)
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias_flat, out,
        N, C, HW,
        float(scaling_factor),
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=8,
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
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        N, IC, IH, IW = x.shape
        OH = (IH - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        OW = (IW - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()
        y = conv_transpose_triton(x, w, b, self.stride, self.padding, OH, OW)
        y = fused_post_conv(y, self.bias, self.scaling_factor)
        return y