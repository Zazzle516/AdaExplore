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
    multiplier,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # one program per (n, oc_tile, hw_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    # output[n, oc, oh, ow] = sum over ic, kh, kw of
    #   x[n, ic, ih, iw] * w[ic, oc, kh, kw]
    # where ih * STRIDE_H = oh + PAD_H - kh, similar for w.

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # ih_num = oh + PAD_H - kh
            ih_num = oh + PAD_H - kh
            iw_num = ow + PAD_W - kw
            # divisible by stride
            ih = ih_num // STRIDE_H
            iw = iw_num // STRIDE_W
            valid = (ih_num >= 0) & (iw_num >= 0) & \
                    ((ih_num % STRIDE_H) == 0) & ((iw_num % STRIDE_W) == 0) & \
                    (ih < IH) & (iw < IW)
            valid = valid & hw_mask

            for ic in range(0, IC):
                # load x[n, ic, ih, iw] -> shape [BLOCK_HW]
                x_idx = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)

                # load w[ic, oc, kh, kw] -> shape [BLOCK_OC]
                w_idx = ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)

                acc += w_val[:, None] * x_val[None, :]

    # add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]
    acc = acc * multiplier

    # store output
    out_idx = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + hw_offs[None, :]
    out_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


@triton.jit
def mean_hw_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    n = pid // C
    c = pid % C

    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for start in range(0, HW, BLOCK):
        idx = start + offs
        mask = idx < HW
        x = tl.load(x_ptr + pid * HW + idx, mask=mask, other=0.0)
        acc += x

    s = tl.sum(acc, axis=0)
    mean_val = s / HW
    tl.store(out_ptr + pid, mean_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()
        bias = self.conv_transpose.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OPH = OPW = self.output_padding

        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        conv_out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_HW = 64
        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_HW))

        conv_transpose_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            SH, SW,
            PH, PW,
            float(self.multiplier),
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=2,
        )

        # Now mean over H, W
        out_mean = torch.empty((N, OC), device=x.device, dtype=x.dtype)
        HW = OH * OW
        BLOCK = 1024
        mean_hw_kernel[(N * OC,)](
            conv_out, out_mean,
            N, OC, HW,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out_mean.view(N, OC, 1, 1)