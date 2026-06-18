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
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
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

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # For ConvTranspose2d: out[n,oc,oh,ow] = sum over ic, kh, kw of
    #   x[n, ic, ih, iw] * w[ic, oc, kh, kw]
    # where ih*stride - pad + kh = oh  =>  ih = (oh + pad - kh) / stride
    # and (oh + pad - kh) % stride == 0
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_num = oh + PAD - kh
            iw_num = ow + PAD - kw
            ih = ih_num // STRIDE
            iw = iw_num // STRIDE
            valid = (ih_num % STRIDE == 0) & (iw_num % STRIDE == 0)
            valid = valid & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            valid = valid & sp_mask

            for ic in range(0, IC):
                # x[n, ic, ih, iw]
                x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)

                # w[ic, oc, kh, kw]
                w_off = ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                acc += w_val[:, None] * x_val[None, :]

    # Add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]

    # clamp(0,1) -> *scale -> clamp(0,1) -> /scale
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * SCALE
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * INV_SCALE

    # Store: out[n, oc, oh, ow]
    out_off = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
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
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = (IH - 1) * self.stride - 2 * self.padding + KH + self.output_padding
        OW = (IW - 1) * self.stride - 2 * self.padding + KW + self.output_padding

        # Fuse the (C,1,1) bias parameter into the conv's per-channel bias
        fused_bias = self.conv_transpose.bias + self.bias.view(-1)
        weight = self.conv_transpose.weight.contiguous()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 64
        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

        conv_transpose2d_kernel[grid](
            x, weight, fused_bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            self.stride, self.padding,
            float(self.scaling_factor),
            float(1.0 / self.scaling_factor),
            BLOCK_OC, BLOCK_SP,
            num_warps=4, num_stages=2,
        )
        return out