import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_fused_kernel(
    x_ptr,           # [N, IC, H, W]
    w_ptr,           # [IC, OC, KH, KW]
    b_ptr,           # [OC]
    bias2_ptr,       # [OC]
    out_ptr,         # [N, OC, OH, OW]
    N, IC, OC,
    H, W,
    OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    scaling_factor,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]

    oh = hw_offs // OW
    ow = hw_offs % OW

    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (OH * OW)

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    # For each kernel position
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # input position: ih = (oh + PAD - kh) / STRIDE, valid if divisible and in range
            ih_num = oh + PAD - kh  # [BLOCK_HW]
            iw_num = ow + PAD - kw
            ih = ih_num // STRIDE
            iw = iw_num // STRIDE
            valid_h = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < H)
            valid_w = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < W)
            valid = valid_h & valid_w & hw_mask  # [BLOCK_HW]

            # Sum over IC
            for ic in range(0, IC):
                # x[pid_n, ic, ih, iw]
                x_idx = pid_n * (IC * H * W) + ic * (H * W) + ih * W + iw
                x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)  # [BLOCK_HW]

                # w[ic, oc, kh, kw]
                w_idx = ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += x_val[:, None] * w_val[None, :]

    # add conv bias
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc += b_val[None, :]

    # add user bias (per-channel)
    bias2_val = tl.load(bias2_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias2_val[None, :]

    # clamp [0,1], scale, clamp [0,1], divide
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * scaling_factor
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc / scaling_factor

    # store: out[pid_n, oc, oh, ow]
    out_idx = pid_n * (OC * OH * OW) + oc_offs[None, :] * (OH * OW) + (oh * OW + ow)[:, None]
    out_mask = oc_mask[None, :] & hw_mask[:, None]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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
        N, IC, H, W = x.shape
        KH = KW = self.kernel_size
        OH = (H - 1) * self.stride - 2 * self.padding + KH + self.output_padding
        OW = (W - 1) * self.stride - 2 * self.padding + KW + self.output_padding

        OC = self.out_channels
        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        conv_bias = self.conv_transpose.bias.contiguous()  # [OC]
        bias2 = self.bias.view(-1).contiguous()  # [OC]

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_HW = 64

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_HW))

        conv_transpose2d_fused_kernel[grid](
            x, weight, conv_bias, bias2, out,
            N, IC, OC,
            H, W,
            OH, OW,
            KH, KW,
            self.stride,
            self.padding,
            float(self.scaling_factor),
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=2,
        )
        return out