import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW'],
)
@triton.jit
def conv_transpose_fused_mean_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    scale,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    ic_offs = tl.arange(0, BLOCK_IC)

    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD_H - kh
        ih = ih_num // STRIDE_H
        ih_valid = (ih_num % STRIDE_H == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD_W - kw
            iw = iw_num // STRIDE_W
            iw_valid = (iw_num % STRIDE_W == 0) & (iw >= 0) & (iw < IW)
            valid = ih_valid & iw_valid & hw_mask  # [BLOCK_HW]

            for ic_start in range(0, IC, BLOCK_IC):
                ic_idx = ic_start + ic_offs
                ic_mask = ic_idx < IC

                # x[n, ic, ih, iw] -> [BLOCK_IC, BLOCK_HW]
                x_off = pid_n * (IC * IH * IW) + ic_idx[:, None] * (IH * IW) + ih[None, :] * IW + iw[None, :]
                x_m = ic_mask[:, None] & valid[None, :]
                x_val = tl.load(x_ptr + x_off, mask=x_m, other=0.0)

                # w[ic, oc, kh, kw] -> [BLOCK_OC, BLOCK_IC]
                w_off = ic_idx[None, :] * (OC * KH * KW) + oc_offs[:, None] * (KH * KW) + kh * KW + kw
                w_m = ic_mask[None, :] & oc_mask[:, None]
                w_val = tl.load(w_ptr + w_off, mask=w_m, other=0.0)

                acc += tl.dot(w_val, x_val, allow_tf32=True)

    # Mask invalid HW lanes, then reduce over HW dim
    full_mask = oc_mask[:, None] & hw_mask[None, :]
    acc = tl.where(full_mask, acc, 0.0)
    partial = tl.sum(acc, axis=1)  # [BLOCK_OC]
    partial = partial * scale

    out_off = pid_n * OC + oc_offs
    tl.atomic_add(out_ptr + out_off, partial, mask=oc_mask)


def conv_transpose2d_fused_mean_triton(x, weight, bias, stride, padding, output_padding, multiplier):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    # Accumulator (N, OC) holding sum_over_HW(conv_no_bias) * (multiplier / (OH*OW))
    acc_out = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

    scale = multiplier / float(OH * OW)

    def grid(meta):
        return (N, triton.cdiv(OC, meta['BLOCK_OC']), triton.cdiv(OH * OW, meta['BLOCK_HW']))

    conv_transpose_fused_mean_kernel[grid](
        x, weight, acc_out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, stride,
        padding, padding,
        scale,
    )

    # Add bias contribution: bias * multiplier (broadcast over batch)
    out = acc_out + (bias.to(acc_out.dtype) * multiplier).unsqueeze(0)
    return out.view(N, OC, 1, 1).to(x.dtype)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()
        bias = self.conv_transpose.bias.contiguous().cuda()

        out = conv_transpose2d_fused_mean_triton(
            x, weight, bias, self.stride, self.padding, self.output_padding, self.multiplier
        )
        return out