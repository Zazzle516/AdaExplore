import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_gather_kernel(
    x_ptr,      # [N, IC, H_in, W_in]
    w_ptr,      # [IC, OC, KH, KW]
    b_ptr,      # [OC]
    out_ptr,    # [N, OC, H_out, W_out]
    N, IC, OC,
    H_in, W_in,
    H_out, W_out,
    add_value,
    multiply_value,
    STRIDE: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    HW = H_out * W_out
    sp_off = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_off < HW

    h_out = sp_off // W_out
    w_out = sp_off % W_out

    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_off < OC

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For transposed conv with padding=0, output_padding=0:
    # out[h, w] = sum over (kh, kw, ic) of x[ic, (h - kh)/stride, (w - kw)/stride] * w[ic, oc, kh, kw]
    # only when (h - kh) % stride == 0 and same for w, and indices in range.
    for kh in tl.static_range(0, KH):
        h_in_num = h_out - kh
        h_valid = (h_in_num >= 0) & ((h_in_num % STRIDE) == 0)
        h_in = h_in_num // STRIDE
        h_valid = h_valid & (h_in < H_in)

        for kw in tl.static_range(0, KW):
            w_in_num = w_out - kw
            w_valid = (w_in_num >= 0) & ((w_in_num % STRIDE) == 0)
            w_in = w_in_num // STRIDE
            w_valid = w_valid & (w_in < W_in)

            valid = h_valid & w_valid & sp_mask

            # x offsets: [BLOCK_SP], for each ic load and accumulate
            x_sp_off = h_in * W_in + w_in  # [BLOCK_SP]
            # base offset per sample for ic=0
            x_base = pid_n * (IC * H_in * W_in) + x_sp_off  # [BLOCK_SP]

            # w offsets per (ic, oc): w_ptr + ic*OC*KH*KW + oc*KH*KW + kh*KW + kw
            w_base = oc_off * (KH * KW) + (kh * KW + kw)  # [BLOCK_OC]

            for ic in range(0, IC):
                x_ptrs = x_ptr + x_base + ic * (H_in * W_in)
                x_vals = tl.load(x_ptrs, mask=valid, other=0.0)  # [BLOCK_SP]

                w_ptrs = w_ptr + ic * (OC * KH * KW) + w_base
                w_vals = tl.load(w_ptrs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += x_vals[:, None] * w_vals[None, :]

    # bias
    b_vals = tl.load(b_ptr + oc_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + b_vals[None, :]

    # epilogue: + add_value, min(., 0), GELU, * multiply_value
    acc = acc + add_value
    acc = tl.minimum(acc, 0.0)
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    g = g * multiply_value

    # store: out[n, oc, h_out, w_out]
    out_base = pid_n * (OC * HW)
    out_off = out_base + oc_off[None, :] * HW + sp_off[:, None]
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, g, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = x.contiguous()
        N, IC, H_in, W_in = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        S = self.stride
        H_out = (H_in - 1) * S + KH
        W_out = (W_in - 1) * S + KW

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        bias = self.conv_transpose.bias.contiguous()      # [OC]

        out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_SP = 64

        grid = (
            N,
            triton.cdiv(OC, BLOCK_OC),
            triton.cdiv(H_out * W_out, BLOCK_SP),
        )

        conv_transpose_gather_kernel[grid](
            x, weight, bias, out,
            N, IC, OC,
            H_in, W_in,
            H_out, W_out,
            self.add_value,
            self.multiply_value,
            STRIDE=S,
            KH=KH,
            KW=KW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )
        return out