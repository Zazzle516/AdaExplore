import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'H_out', 'W_out'],
)
@triton.jit
def conv_transpose2d_fused_kernel(
    x_ptr,       # [N, H_in, W_in, IC] channels-last
    w_ptr,       # [IC, KH, KW, OC] (reshaped from [IC, OC, KH, KW])
    bias_ptr,    # [OC]
    out_ptr,     # [N, H_out, W_out, OC]
    N, IC, OC,
    H_in, W_in, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    add_value: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)
    pid_oc = tl.program_id(2)

    HW_out = H_out * W_out
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]

    sp_mask = sp_offs < HW_out
    oc_mask = oc_offs < OC

    # Decompose sp_offs into (h_out, w_out)
    h_out = sp_offs // W_out
    w_out = sp_offs % W_out

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # Base offset for this n in x: pid_n * H_in * W_in * IC
    x_n_base = pid_n * H_in * W_in * IC
    out_n_base = pid_n * H_out * W_out * OC

    ic_range = tl.arange(0, BLOCK_IC)

    for kh in tl.static_range(0, KH):
        # h_in_num = h_out + PAD_H - kh
        h_in_num = h_out + PAD_H - kh
        h_in = h_in_num // STRIDE_H
        h_valid = (h_in_num - h_in * STRIDE_H == 0) & (h_in >= 0) & (h_in < H_in)

        for kw in tl.static_range(0, KW):
            w_in_num = w_out + PAD_W - kw
            w_in = w_in_num // STRIDE_W
            w_valid = (w_in_num - w_in * STRIDE_W == 0) & (w_in >= 0) & (w_in < W_in)

            spatial_valid = h_valid & w_valid & sp_mask  # [BLOCK_SP]

            # Compute input base offset per sp: x_n_base + (h_in * W_in + w_in) * IC
            x_sp_base = x_n_base + (h_in * W_in + w_in) * IC  # [BLOCK_SP]

            # weight base for (kh, kw): w_ptr + (kh * KW + kw) * OC offset within IC stride
            w_khkw_off = (kh * KW + kw) * OC
            w_base_ptr = w_ptr + w_khkw_off + oc_offs[None, :]

            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + ic_range  # [BLOCK_IC]
                ic_mask = ic_offs < IC

                # Load x tile [BLOCK_SP, BLOCK_IC]
                x_ptrs = x_ptr + x_sp_base[:, None] + ic_offs[None, :]
                x_load_mask = spatial_valid[:, None] & ic_mask[None, :]
                x_tile = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

                # Load w tile [BLOCK_IC, BLOCK_OC]
                w_ptrs = w_base_ptr + ic_offs[:, None] * (KH * KW * OC)
                w_load_mask = ic_mask[:, None] & oc_mask[None, :]
                w_tile = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

                acc += tl.dot(x_tile, w_tile)

    # Epilogue: bias + mish + add_value + hardtanh + scale
    bias = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[None, :]

    # Mish: x * tanh(softplus(x))
    ax = tl.abs(acc)
    sp = tl.maximum(acc, 0.0) + tl.log(1.0 + tl.exp(-ax))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = acc * th
    y = y + add_value
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * scale

    # Store [BLOCK_SP, BLOCK_OC] -> output[n, h_out, w_out, oc]
    out_ptrs = out_ptr + out_n_base + sp_offs[:, None] * OC + oc_offs[None, :]
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptrs, y, mask=store_mask)


def conv_transpose2d_fused(x, weight, bias, stride, padding, output_padding, add_value, scale):
    # x: [N, IC, H_in, W_in]
    # weight: [IC, OC, KH, KW]
    N, IC, H_in, W_in = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    sH, sW = stride if isinstance(stride, tuple) else (stride, stride)
    pH, pW = padding if isinstance(padding, tuple) else (padding, padding)
    opH, opW = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)

    H_out = (H_in - 1) * sH - 2 * pH + KH + opH
    W_out = (W_in - 1) * sW - 2 * pW + KW + opW

    # Convert x to channels-last: [N, H_in, W_in, IC]
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()

    # Reshape weight: [IC, OC, KH, KW] -> [IC, KH, KW, OC]
    w_perm = weight.permute(0, 2, 3, 1).contiguous()

    out_nhwc = torch.empty((N, H_out, W_out, OC), device=x.device, dtype=x.dtype)

    HW_out = H_out * W_out

    grid = lambda meta: (
        N,
        triton.cdiv(HW_out, meta['BLOCK_SP']),
        triton.cdiv(OC, meta['BLOCK_OC']),
    )

    conv_transpose2d_fused_kernel[grid](
        x_nhwc, w_perm, bias, out_nhwc,
        N, IC, OC,
        H_in, W_in, H_out, W_out,
        KH, KW,
        sH, sW,
        pH, pW,
        float(add_value), float(scale),
    )

    # Convert back to NCHW
    out = out_nhwc.permute(0, 3, 1, 2).contiguous()
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        return conv_transpose2d_fused(
            x.contiguous(),
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.add_value,
            self.scale,
        )