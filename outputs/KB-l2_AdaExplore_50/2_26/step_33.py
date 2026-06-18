import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, add_ptr, extra_bias_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wi, stride_wo, stride_wkd, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    STRIDE: tl.constexpr, PAD: tl.constexpr, KSIZE: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    OSP = OD * OH * OW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < OSP

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # iterate over kernel
    for kd in tl.static_range(0, KSIZE):
        id_num = od + PAD - kd
        id_idx = id_num // STRIDE
        valid_d = ((id_num % STRIDE) == 0) & (id_idx >= 0) & (id_idx < ID)
        for kh in tl.static_range(0, KSIZE):
            ih_num = oh + PAD - kh
            ih_idx = ih_num // STRIDE
            valid_h = ((ih_num % STRIDE) == 0) & (ih_idx >= 0) & (ih_idx < IH)
            for kw in tl.static_range(0, KSIZE):
                iw_num = ow + PAD - kw
                iw_idx = iw_num // STRIDE
                valid_w = ((iw_num % STRIDE) == 0) & (iw_idx >= 0) & (iw_idx < IW)
                valid = valid_d & valid_h & valid_w & sp_mask

                # x[n, ic, id_idx, ih_idx, iw_idx] for ic in [0, IC)
                # w[ic, oc, kd, kh, kw]
                x_base = (pid_n * stride_xn
                          + id_idx * stride_xd
                          + ih_idx * stride_xh
                          + iw_idx * stride_xw)  # shape [BLOCK_SP]
                w_base = (oc_offs * stride_wo
                          + kd * stride_wkd
                          + kh * stride_wkh
                          + kw * stride_wkw)  # shape [BLOCK_OC]

                for ic in range(0, IC):
                    x_ptrs = x_ptr + x_base + ic * stride_xc
                    x_vals = tl.load(x_ptrs, mask=valid, other=0.0)  # [BLOCK_SP]
                    w_ptrs = w_ptr + w_base + ic * stride_wi
                    w_vals = tl.load(w_ptrs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                    acc += x_vals[:, None] * w_vals[None, :]

    # add conv bias
    cb = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc += cb[None, :]

    # add add_input[n, oc, od, oh, ow]
    out_base = (pid_n * stride_on
                + od * stride_od
                + oh * stride_oh
                + ow * stride_ow)  # [BLOCK_SP]
    full_mask = sp_mask[:, None] & oc_mask[None, :]
    out_ptrs = out_ptr + out_base[:, None] + oc_offs[None, :] * stride_oc

    add_vals = tl.load(out_ptrs, mask=full_mask, other=0.0)  # we'll load from add_ptr instead
    # actually load from add_ptr
    add_ptrs = add_ptr + out_base[:, None] + oc_offs[None, :] * stride_oc
    add_vals = tl.load(add_ptrs, mask=full_mask, other=0.0)
    v = acc + add_vals

    # extra bias is broadcast over (1, OC, 1, 1, 1) - shape (OC,)
    eb = tl.load(extra_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    v = v + eb[None, :]

    # hardswish: v * v * relu6(v+3)/6
    relu6 = tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0)
    hs = v * relu6 * (1.0 / 6.0)
    out_v = v * hs

    tl.store(out_ptrs, out_v, mask=full_mask)


def conv_transpose3d_fused(x, weight, conv_bias, add_input, extra_bias,
                            stride, padding, output_padding, kernel_size):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    OD = (ID - 1) * stride - 2 * padding + KD + output_padding
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 64

    OSP = OD * OH * OW
    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OSP, BLOCK_SP))

    conv_transpose3d_fused_kernel[grid](
        x, weight, conv_bias, add_input, extra_bias, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        STRIDE=stride, PAD=padding, KSIZE=kernel_size,
        BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def forward(self, x, add_input):
        x = x.contiguous()
        add_input = add_input.contiguous()
        w = self.conv_transpose.weight.contiguous()
        cb = self.conv_transpose.bias.contiguous()
        eb = self.bias.view(-1).contiguous()
        return conv_transpose3d_fused(
            x, w, cb, add_input, eb,
            self.stride, self.padding, self.output_padding, self.kernel_size,
        )