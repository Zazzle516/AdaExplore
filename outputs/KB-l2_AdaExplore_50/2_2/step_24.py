import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_epilogue_kernel(
    x_ptr, w_ptr, conv_bias_ptr, extra_bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PADDING: tl.constexpr,
    inv_scale: tl.float32,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    oh = sp_offs // OW
    ow = sp_offs % OW

    # accumulator [BLOCK_SP, BLOCK_OC]
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # Iterate over input channels and kernel positions
    for ic in range(0, IC):
        for kh in range(0, KH):
            for kw in range(0, KW):
                # ih_num = oh + padding - kh ; must be divisible by stride
                ih_num = oh + PADDING - kh
                iw_num = ow + PADDING - kw
                ih = ih_num // STRIDE
                iw = iw_num // STRIDE
                valid = (ih_num % STRIDE == 0) & (iw_num % STRIDE == 0) & \
                        (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)

                # Load x[n, ic, ih, iw] -> [BLOCK_SP]
                x_idx = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                x_val = tl.load(x_ptr + x_idx, mask=sp_mask & valid, other=0.0)

                # Load weight[ic, oc_offs, kh, kw] -> [BLOCK_OC]
                w_idx = ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)

                # outer product
                acc += x_val[:, None] * w_val[None, :]

    # Add conv bias [OC]
    cb = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += cb[None, :]

    # Add extra bias [OC, 1, 1] -> per channel
    eb = tl.load(extra_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += eb[None, :]

    # clamp [0,1], *scale, clamp[0,1], /scale -> equivalent to min(acc, inv_scale) clamped to [0, inv_scale]? Let's just do it:
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * (1.0 / inv_scale)  # inv_scale here is actually scaling_factor; we'll pass scaling_factor
    # Wait, let me re-do with proper var

    # store
    out_idx = pid_n * (OC * OH * OW) + oc_offs[None, :] * (OH * OW) + sp_offs[:, None]
    mask_out = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=mask_out)


@triton.jit
def conv_transpose2d_epilogue_kernel_v2(
    x_ptr, w_ptr, conv_bias_ptr, extra_bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PADDING: tl.constexpr,
    scaling_factor: tl.float32,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    oh = sp_offs // OW
    ow = sp_offs % OW

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    for ic in range(0, IC):
        for kh in range(0, KH):
            for kw in range(0, KW):
                ih_num = oh + PADDING - kh
                iw_num = ow + PADDING - kw
                ih = ih_num // STRIDE
                iw = iw_num // STRIDE
                valid = (ih_num % STRIDE == 0) & (iw_num % STRIDE == 0) & \
                        (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)

                x_idx = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                x_val = tl.load(x_ptr + x_idx, mask=sp_mask & valid, other=0.0)

                w_idx = ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)

                acc += x_val[:, None] * w_val[None, :]

    cb = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += cb[None, :]

    eb = tl.load(extra_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += eb[None, :]

    # clamp [0,1]
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    # multiply by scaling factor
    acc = acc * scaling_factor
    # clamp [0,1]
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    # divide by scaling factor
    acc = acc / scaling_factor

    out_idx = pid_n * (OC * OH * OW) + oc_offs[None, :] * (OH * OW) + sp_offs[:, None]
    mask_out = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=mask_out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        OC = self.out_channels
        OH = (IH - 1) * self.stride - 2 * self.padding + KH + self.output_padding
        OW = (IW - 1) * self.stride - 2 * self.padding + KW + self.output_padding

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        conv_bias = self.conv_transpose.bias.contiguous()  # [OC]
        extra_bias = self.bias.view(-1).contiguous()  # [OC]

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 64

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

        conv_transpose2d_epilogue_kernel_v2[grid](
            x, weight, conv_bias, extra_bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            self.stride, self.padding,
            float(self.scaling_factor),
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )
        return out