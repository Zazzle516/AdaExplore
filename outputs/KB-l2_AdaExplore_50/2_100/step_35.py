import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_gather_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    KD, KH, KW,
    stride, padding,
    min_value, inv_divisor,
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

    d_o = sp_offs // HW_out
    rem = sp_offs % HW_out
    h_o = rem // W_out
    w_o = rem % W_out

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For each output position, iterate over kernel positions
    # input position: (d_o + padding - kd) must be divisible by stride
    # input index: (d_o + padding - kd) / stride, must be in [0, D_in)
    for kd in range(KD):
        d_num = d_o + padding - kd
        d_in = d_num // stride
        d_valid = (d_num >= 0) & ((d_num % stride) == 0) & (d_in < D_in) & (d_in >= 0)
        for kh in range(KH):
            h_num = h_o + padding - kh
            h_in = h_num // stride
            h_valid = (h_num >= 0) & ((h_num % stride) == 0) & (h_in < H_in) & (h_in >= 0)
            for kw in range(KW):
                w_num = w_o + padding - kw
                w_in = w_num // stride
                w_valid = (w_num >= 0) & ((w_num % stride) == 0) & (w_in < W_in) & (w_in >= 0)

                spatial_valid = d_valid & h_valid & w_valid & sp_mask  # [BLOCK_SP]

                # input offset for each spatial block element
                in_spatial_off = d_in * (H_in * W_in) + h_in * W_in + w_in  # [BLOCK_SP]

                # Loop over input channels
                for ic in range(IC):
                    # x[n, ic, d_in, h_in, w_in]
                    x_off = pid_n * (IC * D_in * H_in * W_in) + ic * (D_in * H_in * W_in) + in_spatial_off
                    x_val = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)  # [BLOCK_SP]

                    # weight[ic, oc, kd, kh, kw]
                    w_off = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += x_val[:, None] * w_val[None, :]

    # Add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[None, :]

    # Clamp + divide
    acc = tl.where(acc < min_value, min_value, acc)
    acc = acc * inv_divisor

    # Store: out[n, oc, d_o, h_o, w_o]
    out_off = pid_n * (OC * DHW_out) + oc_offs[None, :] * DHW_out + sp_offs[:, None]
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


@triton.jit
def conv_transpose3d_gather_kernel_v2(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    KD, KH, KW,
    stride, padding,
    min_value, inv_divisor,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    HW_out = H_out * W_out
    DHW_out = D_out * HW_out
    HW_in = H_in * W_in
    DHW_in = D_in * HW_in
    KDHW = KD * KH * KW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < DHW_out

    d_o = sp_offs // HW_out
    rem = sp_offs % HW_out
    h_o = rem // W_out
    w_o = rem % W_out

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    ic_range = tl.arange(0, BLOCK_IC)

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    for kd in range(KD):
        d_num = d_o + padding - kd
        d_in = d_num // stride
        d_valid = (d_num >= 0) & ((d_num % stride) == 0) & (d_in < D_in)
        for kh in range(KH):
            h_num = h_o + padding - kh
            h_in = h_num // stride
            h_valid = (h_num >= 0) & ((h_num % stride) == 0) & (h_in < H_in)
            for kw in range(KW):
                w_num = w_o + padding - kw
                w_in = w_num // stride
                w_valid = (w_num >= 0) & ((w_num % stride) == 0) & (w_in < W_in)

                spatial_valid = d_valid & h_valid & w_valid & sp_mask

                in_spatial_off = d_in * HW_in + h_in * W_in + w_in

                kk = kd * (KH * KW) + kh * KW + kw

                for ic_start in range(0, IC, BLOCK_IC):
                    ic_offs = ic_start + ic_range
                    ic_mask = ic_offs < IC

                    # Load x[n, ic_offs, in_spatial_off]: shape [BLOCK_SP, BLOCK_IC]
                    x_off = pid_n * (IC * DHW_in) + ic_offs[None, :] * DHW_in + in_spatial_off[:, None]
                    x_mask = spatial_valid[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                    # Load weight[ic_offs, oc_offs, kk]: shape [BLOCK_IC, BLOCK_OC]
                    w_off = ic_offs[:, None] * (OC * KDHW) + oc_offs[None, :] * KDHW + kk
                    w_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                    acc += tl.dot(x_vals, w_vals)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.where(acc < min_value, min_value, acc)
    acc = acc * inv_divisor

    out_off = pid_n * (OC * DHW_out) + oc_offs[None, :] * DHW_out + sp_offs[:, None]
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv_transpose3d_fused(x, weight, bias, stride, padding, min_value, divisor):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, IC, D_in, H_in, W_in = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    D_out = (D_in - 1) * stride - 2 * padding + KD
    H_out = (H_in - 1) * stride - 2 * padding + KH
    W_out = (W_in - 1) * stride - 2 * padding + KW

    out = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    DHW_out = D_out * H_out * W_out
    grid = lambda meta: (
        (DHW_out + meta['BLOCK_SP'] - 1) // meta['BLOCK_SP'],
        (OC + meta['BLOCK_OC'] - 1) // meta['BLOCK_OC'],
        N,
    )

    conv_transpose3d_gather_kernel_v2[grid](
        x, weight, bias, out,
        N, IC, OC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        KD, KH, KW,
        stride, padding,
        float(min_value), float(1.0 / divisor),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        return conv_transpose3d_fused(x, weight, bias, self.stride, self.padding, self.min_value, self.divisor)