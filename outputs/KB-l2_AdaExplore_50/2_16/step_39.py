import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


# Output-stationary transposed conv kernel.
# Layout: input is (N, IC, H_in, W_in) contiguous (NCHW).
#         weight is (IC, OC, KH, KW) contiguous as in PyTorch ConvTranspose2d.
#         output is (N, OC, H_out, W_out) contiguous (NCHW).
# stride=2, padding=1, output_padding=1, kernel=3 specifically.
# H_out = (H_in - 1) * 2 - 2 + 3 + 1 = 2*H_in
# W_out = 2*W_in
#
# For an output position (h_out, w_out):
#   h_in_eff = h_out + padding - kh   (must be divisible by stride)
#   h_in     = h_in_eff / stride
#   similarly for w. valid kh values depend on (h_out + padding) % stride.
#
# We tile over (N, OC tile, spatial tile). Each program handles BLOCK_M output
# channels and BLOCK_N output spatial positions for one batch element.

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    ],
    key=["IC", "OC", "H_in", "W_in"],
)
@triton.jit
def conv_transpose2d_fused_kernel(
    x_ptr,        # (N, IC, H_in, W_in)
    w_ptr,        # (IC, OC, KH, KW)
    b_ptr,        # (OC,)
    out_ptr,      # (N, OC, H_out, W_out)
    N, IC, OC,
    H_in, W_in,
    H_out, W_out,
    ADD_VALUE: tl.constexpr,
    SCALE: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,   # OC tile
    BLOCK_N: tl.constexpr,   # spatial tile (H_out*W_out)
    BLOCK_K: tl.constexpr,   # IC tile
):
    pid_n = tl.program_id(0)        # batch index
    pid_m = tl.program_id(1)        # OC tile
    pid_s = tl.program_id(2)        # spatial tile

    # OC indices for this tile
    offs_oc = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    mask_oc = offs_oc < OC

    # Spatial output indices for this tile
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)    # [BLOCK_N]
    mask_s = offs_s < (H_out * W_out)
    h_out = offs_s // W_out
    w_out = offs_s % W_out

    # For each output pixel, compute h_in_eff = h_out + PADDING (then subtract kh)
    h_in_eff = h_out + PADDING                          # [BLOCK_N]
    w_in_eff = w_out + PADDING

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over input channel tiles
    # Pre-compute pointer bases
    # x layout: x[n, ic, hi, wi] -> n*IC*Hin*Win + ic*Hin*Win + hi*Win + wi
    # w layout: w[ic, oc, kh, kw] -> ic*OC*KH*KW + oc*KH*KW + kh*KW + kw
    n_offset = pid_n * IC * H_in * W_in

    # Iterate kernel positions explicitly (KH*KW small)
    # For each kh, kw: determine validity per output pixel
    # h_in_num = h_in_eff - kh; valid if (h_in_num % STRIDE == 0) and 0 <= h_in_num/STRIDE < H_in
    # similarly for w.

    for kh in tl.static_range(0, KH):
        h_in_num = h_in_eff - kh                        # [BLOCK_N]
        h_in = h_in_num // STRIDE
        h_valid = ((h_in_num % STRIDE) == 0) & (h_in >= 0) & (h_in < H_in)
        for kw in tl.static_range(0, KW):
            w_in_num = w_in_eff - kw
            w_in = w_in_num // STRIDE
            w_valid = ((w_in_num % STRIDE) == 0) & (w_in >= 0) & (w_in < W_in)
            valid = h_valid & w_valid                   # [BLOCK_N]

            # Compute base spatial offset into x for this (kh, kw)
            x_spatial = h_in * W_in + w_in              # [BLOCK_N]

            # Loop over IC tiles
            for ic_start in range(0, IC, BLOCK_K):
                offs_ic = ic_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]
                mask_ic = offs_ic < IC

                # Load x tile: shape [BLOCK_K, BLOCK_N]
                # x_ptrs[k, n] = n_offset + offs_ic[k]*Hin*Win + x_spatial[n]
                x_ptrs = (x_ptr + n_offset
                          + offs_ic[:, None] * (H_in * W_in)
                          + x_spatial[None, :])
                x_mask = mask_ic[:, None] & valid[None, :] & mask_s[None, :]
                x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)   # [BLOCK_K, BLOCK_N]

                # Load w tile: shape [BLOCK_K, BLOCK_M]
                # w[ic, oc, kh, kw]
                w_ptrs = (w_ptr
                          + offs_ic[:, None] * (OC * KH * KW)
                          + offs_oc[None, :] * (KH * KW)
                          + kh * KW + kw)
                w_mask = mask_ic[:, None] & mask_oc[None, :]
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)   # [BLOCK_K, BLOCK_M]

                # acc[m, n] += sum_k w_tile[k, m] * x_tile[k, n]
                acc += tl.dot(tl.trans(w_tile), x_tile, allow_tf32=True)

    # Add bias
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)        # [BLOCK_M]
    acc = acc + bias[:, None]

    # Mish: x * tanh(softplus(x))
    # stable softplus
    sp = tl.log(1.0 + tl.exp(-tl.abs(acc))) + tl.maximum(acc, 0.0)
    e2 = tl.exp(2.0 * sp)
    tanh_sp = 1.0 - 2.0 / (e2 + 1.0)
    y = acc * tanh_sp
    y = y + ADD_VALUE
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * SCALE

    # Store: out[n, oc, h_out, w_out]
    out_offset = pid_n * OC * H_out * W_out
    out_ptrs = (out_ptr + out_offset
                + offs_oc[:, None] * (H_out * W_out)
                + offs_s[None, :])
    out_mask = mask_oc[:, None] & mask_s[None, :]
    tl.store(out_ptrs, y, mask=out_mask)


def conv_transpose2d_fused(x, weight, bias, stride, padding, output_padding,
                           kernel_size, add_value, scale):
    N, IC, H_in, W_in = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w
    H_out = (H_in - 1) * stride - 2 * padding + KH + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + KW + output_padding

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        N,
        triton.cdiv(OC, META["BLOCK_M"]),
        triton.cdiv(H_out * W_out, META["BLOCK_N"]),
    )

    conv_transpose2d_fused_kernel[grid](
        x, weight, bias, out,
        N, IC, OC,
        H_in, W_in, H_out, W_out,
        float(add_value), float(scale),
        stride, padding, KH, KW,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, add_value, scale):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = x.contiguous()
        return conv_transpose2d_fused(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.kernel_size,
            self.add_value,
            self.scale,
        )