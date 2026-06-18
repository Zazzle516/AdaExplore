import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # program: (n, oc_block, hw_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    # for transposed conv: out[oh, ow] = sum over (ic, kh, kw) where
    # ih = (oh + PAD_H - kh) / STRIDE_H, iw similarly, divisible by stride
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

            for ic in range(0, IC):
                # load x[n, ic, ih, iw] -> [BLOCK_HW]
                x_offset = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)

                # load w[ic, oc, kh, kw] -> [BLOCK_OC]
                w_offset = ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_offset, mask=oc_mask, other=0.0)

                acc += w_val[:, None] * x_val[None, :]

    # add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    # store
    out_offset = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + hw_offs[None, :]
    out_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


def conv_transpose2d_triton(x, weight, bias, stride, padding, output_padding):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_HW = 64

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_HW))

    conv_transpose_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, stride,
        padding, padding,
        BLOCK_OC=BLOCK_OC,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
    )
    return out


@triton.jit
def mean_mul_kernel(
    x_ptr, out_ptr,
    N, OC, HW,
    multiplier,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)  # n * OC + oc
    offs = tl.arange(0, BLOCK_HW)

    x_base = pid * HW
    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    for start in range(0, HW, BLOCK_HW):
        idx = start + offs
        mask = idx < HW
        v = tl.load(x_ptr + x_base + idx, mask=mask, other=0.0)
        acc += v

    total = tl.sum(acc, axis=0)
    mean = total / HW * multiplier
    tl.store(out_ptr + pid, mean)


def mean_mul_triton(x, multiplier):
    N, OC, H, W = x.shape
    HW = H * W
    out = torch.empty((N, OC, 1, 1), device=x.device, dtype=x.dtype)
    grid = (N * OC,)
    BLOCK_HW = 1024
    mean_mul_kernel[grid](x, out, N, OC, HW, multiplier, BLOCK_HW=BLOCK_HW, num_warps=4)
    return out


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

        y = conv_transpose2d_triton(x, weight, bias, self.stride, self.padding, self.output_padding)
        out = mean_mul_triton(y, self.multiplier)
        return out