import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# Output dimensions: H_out = (H_in - 1) * stride + kernel_size  (no padding, no output_padding)
# For our case: H_in=64, stride=2, kernel=4 -> H_out = 63*2 + 4 = 130
# W_out = 130 similarly.

# Gather-style kernel: one program per (N, OC tile, H_out tile, W_out tile).
# For each output pixel (h_out, w_out), iterate over kernel positions (kh, kw) such that
# (h_out - kh) is divisible by stride and h_in = (h_out - kh) / stride is in [0, H_in).
# Sum over IC.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 32}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'OC', 'H_in', 'W_in'],
)
@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,       # [N, IC, H_in, W_in]
    w_ptr,       # [IC, OC, KH, KW]
    b_ptr,       # [OC]
    out_ptr,     # [N, OC, H_out, W_out]
    N, IC, OC,
    H_in, W_in,
    H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    add_value: tl.constexpr,
    multiply_value: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    HW_out = H_out * W_out
    hw_start = pid_hw * BLOCK_HW
    hw_offs = hw_start + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]
    hw_mask = hw_offs < HW_out

    h_out = hw_offs // W_out
    w_out = hw_offs % W_out

    oc_start = pid_oc * BLOCK_OC
    oc_offs = oc_start + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    # Iterate over kernel positions
    for kh in tl.static_range(0, KH):
        # h_in_num = h_out - kh; must be >= 0 and divisible by stride and < H_in*stride
        h_in_num = h_out - kh  # [BLOCK_HW]
        h_in = h_in_num // STRIDE
        h_valid = (h_in_num >= 0) & ((h_in_num % STRIDE) == 0) & (h_in < H_in) & (h_in >= 0)

        for kw in tl.static_range(0, KW):
            w_in_num = w_out - kw
            w_in = w_in_num // STRIDE
            w_valid = (w_in_num >= 0) & ((w_in_num % STRIDE) == 0) & (w_in < W_in) & (w_in >= 0)

            valid = h_valid & w_valid & hw_mask  # [BLOCK_HW]

            # Sum over IC: acc[hw, oc] += sum_ic( x[n, ic, h_in, w_in] * w[ic, oc, kh, kw] )
            # Load x slice: [BLOCK_HW, IC]
            # x offset: pid_n * IC * H_in * W_in + ic * H_in * W_in + h_in * W_in + w_in
            # w offset: ic * OC * KH * KW + oc * KH * KW + kh * KW + kw
            spatial_off = h_in * W_in + w_in  # [BLOCK_HW]
            x_base = pid_n * IC * H_in * W_in + spatial_off  # [BLOCK_HW]
            w_base = oc_offs * KH * KW + kh * KW + kw  # [BLOCK_OC]

            # Accumulate over IC in chunks
            for ic in range(0, IC):
                x_off = x_base + ic * H_in * W_in  # [BLOCK_HW]
                w_off = ic * OC * KH * KW + w_base  # [BLOCK_OC]

                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)  # [BLOCK_HW]
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += x_val[:, None] * w_val[None, :]

    # Add bias
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + b_val[None, :]

    # Add value
    acc = acc + add_value
    # min(x, 0)
    acc = tl.minimum(acc, 0.0)
    # GELU
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    # Multiply
    acc = acc * multiply_value

    # Store: out[n, oc, h_out, w_out]
    out_off = pid_n * OC * HW_out + oc_offs[None, :] * HW_out + hw_offs[:, None]
    store_mask = oc_mask[None, :] & hw_mask[:, None]
    tl.store(out_ptr + out_off, acc, mask=store_mask)


# Better version: use BLOCK_IC tiling to amortize loads
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 32, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 32, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 32, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 32, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 16, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'OC', 'H_in', 'W_in'],
)
@triton.jit
def conv_transpose_fused_kernel_v2(
    x_ptr,       # [N, IC, H_in, W_in]
    w_ptr,       # [IC, OC, KH, KW]  flattened
    b_ptr,       # [OC]
    out_ptr,     # [N, OC, H_out, W_out]
    N, IC, OC,
    H_in, W_in,
    H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    add_value: tl.constexpr,
    multiply_value: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    HW_out = H_out * W_out
    HW_in = H_in * W_in
    hw_start = pid_hw * BLOCK_HW
    hw_offs = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW_out

    h_out = hw_offs // W_out
    w_out = hw_offs % W_out

    oc_start = pid_oc * BLOCK_OC
    oc_offs = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    ic_offs = tl.arange(0, BLOCK_IC)

    for kh in tl.static_range(0, KH):
        h_in_num = h_out - kh
        h_in = h_in_num // STRIDE
        h_valid = (h_in_num >= 0) & ((h_in_num % STRIDE) == 0) & (h_in < H_in) & (h_in >= 0)

        for kw in tl.static_range(0, KW):
            w_in_num = w_out - kw
            w_in = w_in_num // STRIDE
            w_valid = (w_in_num >= 0) & ((w_in_num % STRIDE) == 0) & (w_in < W_in) & (w_in >= 0)

            valid = h_valid & w_valid & hw_mask  # [BLOCK_HW]
            spatial_off = h_in * W_in + w_in  # [BLOCK_HW]

            for ic_start in range(0, IC, BLOCK_IC):
                ic_idx = ic_start + ic_offs  # [BLOCK_IC]
                ic_mask = ic_idx < IC

                # x: [BLOCK_HW, BLOCK_IC]
                x_off = pid_n * IC * HW_in + ic_idx[None, :] * HW_in + spatial_off[:, None]
                x_mask = valid[:, None] & ic_mask[None, :]
                x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                # w: [BLOCK_IC, BLOCK_OC]
                w_off = ic_idx[:, None] * OC * KH * KW + oc_offs[None, :] * KH * KW + kh * KW + kw
                w_mask_full = ic_mask[:, None] & oc_mask[None, :]
                w_val = tl.load(w_ptr + w_off, mask=w_mask_full, other=0.0)

                acc += tl.dot(x_val, w_val, allow_tf32=True)

    # Bias
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_val[None, :]

    # add_value
    acc = acc + add_value
    # min(x, 0)
    acc = tl.minimum(acc, 0.0)
    # GELU exact
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    # multiply
    acc = acc * multiply_value

    out_off = pid_n * OC * HW_out + oc_offs[None, :] * HW_out + hw_offs[:, None]
    store_mask = oc_mask[None, :] & hw_mask[:, None]
    tl.store(out_ptr + out_off, acc, mask=store_mask)


def conv_transpose_fused(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                          stride: int, kernel_size: int,
                          add_value: float, multiply_value: float) -> torch.Tensor:
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, IC, H_in, W_in = x.shape
    _, OC, KH, KW = weight.shape
    H_out = (H_in - 1) * stride + KH
    W_out = (W_in - 1) * stride + KW

    out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

    HW_out = H_out * W_out

    grid = lambda meta: (
        N,
        triton.cdiv(OC, meta['BLOCK_OC']),
        triton.cdiv(HW_out, meta['BLOCK_HW']),
    )

    conv_transpose_fused_kernel_v2[grid](
        x, weight, bias, out,
        N, IC, OC,
        H_in, W_in,
        H_out, W_out,
        KH=KH, KW=KW,
        STRIDE=stride,
        add_value=float(add_value),
        multiply_value=float(multiply_value),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda()
        return conv_transpose_fused(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.kernel_size,
            self.add_value,
            self.multiply_value,
        )