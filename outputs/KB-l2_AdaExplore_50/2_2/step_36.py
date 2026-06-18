import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_fused_kernel(
    x_ptr,        # [N, IC, H, W]
    w_ptr,        # [IC, OC, KH, KW]
    bias_ptr,     # [OC]
    extra_bias_ptr,  # [OC]
    out_ptr,      # [N, OC, OH, OW]
    N, IC, OC,
    H, W,
    OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    stride: tl.constexpr, pad: tl.constexpr,
    scaling_factor,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]

    oh = hw_offs // OW
    ow = hw_offs % OW
    hw_mask = hw_offs < (OH * OW)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    # For ConvTranspose2d:
    # out[oh, ow] = sum_{ic, kh, kw} x[ic, ih, iw] * w[ic, oc, kh, kw]
    # where ih = (oh + pad - kh) / stride, with divisibility check
    # and iw = (ow + pad - kw) / stride, similar check

    for kh in tl.static_range(0, KH):
        ih_num = oh + pad - kh
        ih = ih_num // stride
        ih_valid = (ih_num % stride == 0) & (ih >= 0) & (ih < H)
        for kw in tl.static_range(0, KW):
            iw_num = ow + pad - kw
            iw = iw_num // stride
            iw_valid = (iw_num % stride == 0) & (iw >= 0) & (iw < W)
            spatial_valid = ih_valid & iw_valid & hw_mask  # [BLOCK_HW]

            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                # Load x[n, ic, ih, iw]: shape [BLOCK_HW, BLOCK_IC]
                x_ptrs = (x_ptr
                          + pid_n * IC * H * W
                          + ic_offs[None, :] * H * W
                          + ih[:, None] * W
                          + iw[:, None])
                x_load_mask = spatial_valid[:, None] & ic_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)  # [BLOCK_HW, BLOCK_IC]

                # Load w[ic, oc, kh, kw]: shape [BLOCK_IC, BLOCK_OC]
                w_ptrs = (w_ptr
                          + ic_offs[:, None] * (OC * KH * KW)
                          + oc_offs[None, :] * (KH * KW)
                          + kh * KW
                          + kw)
                w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]

                acc += tl.dot(x_vals, w_vals)

    # Add conv bias + extra bias
    bias_vals = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    extra_bias_vals = tl.load(extra_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias_vals[None, :] + extra_bias_vals[None, :]

    # Apply: clamp(0,1), *scale, clamp(0,1), /scale
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * scaling_factor
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc / scaling_factor

    # Store output: [N, OC, OH, OW]
    out_ptrs = (out_ptr
                + pid_n * OC * OH * OW
                + oc_offs[None, :] * (OH * OW)
                + hw_offs[:, None])
    store_mask = hw_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


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
        x = x.contiguous().cuda()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        stride = self.stride
        pad = self.padding
        opad = self.output_padding

        OH = (H - 1) * stride - 2 * pad + KH + opad
        OW = (W - 1) * stride - 2 * pad + KW + opad

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        conv_bias = self.conv_transpose.bias.contiguous()  # [OC]
        extra_bias = self.bias.view(-1).contiguous()  # [OC]

        BLOCK_OC = 32
        BLOCK_HW = 64
        BLOCK_IC = 32

        grid = (
            N,
            triton.cdiv(OC, BLOCK_OC),
            triton.cdiv(OH * OW, BLOCK_HW),
        )

        conv_transpose2d_fused_kernel[grid](
            x, weight, conv_bias, extra_bias, out,
            N, IC, OC,
            H, W,
            OH, OW,
            KH, KW,
            stride, pad,
            float(self.scaling_factor),
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            BLOCK_IC=BLOCK_IC,
            num_warps=4,
            num_stages=2,
        )
        return out