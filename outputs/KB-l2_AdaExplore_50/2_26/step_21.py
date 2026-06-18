import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Direct gather-based ConvTranspose3d with fused add + hardswish epilogue.
# Each program computes a tile of output for one (n, oc_block, spatial_block).
# Output[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw} input[n, ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
# where id = (od + pad - kd)/stride, valid when (od+pad-kd) % stride == 0 and 0 <= id < D_in

@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, add_ptr, out_ptr,
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    HW_out = H_out * W_out
    DHW_out = D_out * HW_out

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < DHW_out

    od = sp_offs // HW_out
    rem = sp_offs % HW_out
    oh = rem // W_out
    ow = rem % W_out

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Precompute base offsets
    # input layout: [N, IC, D_in, H_in, W_in]
    # weight layout: [IC, OC, KD, KH, KW]
    # output layout: [N, OC, D_out, H_out, W_out]

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # iterate over kernel positions and input channels
    for kd in tl.static_range(0, KD):
        id_ = od + PAD - kd
        id_valid = (id_ % STRIDE) == 0
        id_div = id_ // STRIDE
        d_valid = id_valid & (id_div >= 0) & (id_div < D_in)
        for kh in tl.static_range(0, KH):
            ih_ = oh + PAD - kh
            ih_valid = (ih_ % STRIDE) == 0
            ih_div = ih_ // STRIDE
            h_valid = ih_valid & (ih_div >= 0) & (ih_div < H_in)
            for kw in tl.static_range(0, KW):
                iw_ = ow + PAD - kw
                iw_valid = (iw_ % STRIDE) == 0
                iw_div = iw_ // STRIDE
                w_valid = iw_valid & (iw_div >= 0) & (iw_div < W_in)
                spatial_valid = d_valid & h_valid & w_valid & sp_mask

                # input offset for [n, ic, id_div, ih_div, iw_div] - we'll add ic*D_in*H_in*W_in inside loop
                in_spatial_off = id_div * (H_in * W_in) + ih_div * W_in + iw_div  # [BLOCK_SP]
                in_base = pid_n * (IC * D_in * H_in * W_in) + in_spatial_off  # [BLOCK_SP]

                # weight offset for [ic, oc, kd, kh, kw]: ic*OC*KD*KH*KW + oc*KD*KH*KW + (kd*KH+kh)*KW + kw
                k_off = (kd * KH + kh) * KW + kw
                w_base = oc_offs * (KD * KH * KW) + k_off  # [BLOCK_OC]

                for ic in range(0, IC):
                    x_off = in_base + ic * (D_in * H_in * W_in)
                    x_val = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)  # [BLOCK_SP]

                    w_off = ic * (OC * KD * KH * KW) + w_base
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += x_val[:, None] * w_val[None, :]

    # add conv bias
    cb = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += cb[None, :]

    # add the add_input tensor
    # output index: n*OC*DHW + oc*DHW + sp
    out_off = pid_n * (OC * DHW_out) + oc_offs[None, :] * DHW_out + sp_offs[:, None]
    full_mask = sp_mask[:, None] & oc_mask[None, :]

    add_val = tl.load(add_ptr + out_off, mask=full_mask, other=0.0)
    v = acc + add_val

    # hardswish(v) = v * relu6(v+3)/6 ; output = v * hardswish(v) = v^2 * relu6(v+3)/6
    hs = v * tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0) * (1.0 / 6.0)
    out_val = v * hs

    tl.store(out_ptr + out_off, out_val, mask=full_mask)


def conv_transpose3d_fused(x, weight, conv_bias, add_input,
                            stride, padding, output_padding,
                            D_out, H_out, W_out):
    N, IC, D_in, H_in, W_in = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    x = x.contiguous()
    weight = weight.contiguous()
    add_input = add_input.contiguous()

    out = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 128

    DHW_out = D_out * H_out * W_out
    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(DHW_out, BLOCK_SP))

    conv_transpose3d_fused_kernel[grid](
        x, weight, conv_bias, add_input, out,
        N, IC, OC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        KD, KH, KW,
        stride, padding,
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, add_input):
        N, IC, D_in, H_in, W_in = x.shape
        s = self.stride
        p = self.padding
        op = self.output_padding
        k = self.kernel_size
        D_out = (D_in - 1) * s - 2 * p + k + op
        H_out = (H_in - 1) * s - 2 * p + k + op
        W_out = (W_in - 1) * s - 2 * p + k + op

        return conv_transpose3d_fused(
            x, self.conv_transpose.weight, self.conv_transpose.bias, add_input,
            s, p, op, D_out, H_out, W_out,
        )