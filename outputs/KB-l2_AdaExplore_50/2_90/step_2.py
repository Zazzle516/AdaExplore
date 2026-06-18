import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Implicit GEMM Conv3d with fused epilogue (bias + leaky_relu + sum_tensor + clamp + gelu)
# Layout: input  NDHWC  (channels last), weight (OC, kT, kH, kW, IC), output NDHWC
# One program per (N, OC_tile, spatial_tile) where spatial = D_out * H_out * W_out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'D_out', 'H_out', 'W_out', 'KT', 'KH', 'KW'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr,         # input NDHWC: (N, D_in, H_in, W_in, IC)
    w_ptr,         # weight (OC, KT, KH, KW, IC)
    b_ptr,         # bias (OC,)  -- folded with sum_tensor already if present
    s_ptr,         # sum_tensor (OC,)  -- already added to bias OR separate; here folded
    out_ptr,       # output NDHWC: (N, D_out, H_out, W_out, OC)
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    KT: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    stride_x_n, stride_x_d, stride_x_h, stride_x_w,  # IC stride is 1
    stride_w_oc,  # weight: OC stride; inner is (KT*KH*KW*IC) contiguous
    stride_o_n, stride_o_d, stride_o_h, stride_o_w,  # OC stride is 1
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
    BLOCK_K: tl.constexpr,  # K tile (IC)
):
    pid_n = tl.program_id(0)              # batch
    pid_m = tl.program_id(1)              # OC tile
    pid_s = tl.program_id(2)              # spatial tile

    HW_out = H_out * W_out
    DHW_out = D_out * HW_out

    # spatial offsets within this tile
    s_off = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)
    s_mask = s_off < DHW_out

    d_idx = s_off // HW_out
    rem = s_off % HW_out
    h_idx = rem // W_out
    w_idx = rem % W_out

    # OC offsets
    oc_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    oc_mask = oc_off < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kT, kH, kW, and IC (in BLOCK_K chunks)
    # K-dimension order: kt, kh, kw, ic  (matches weight layout (OC, KT, KH, KW, IC))
    # Number of IC blocks
    IC_BLOCKS = (IC + BLOCK_K - 1) // BLOCK_K

    for kt in tl.static_range(0, KT):
        d_in = d_idx + kt  # padding=0, stride=1
        for kh in tl.static_range(0, KH):
            h_in = h_idx + kh
            for kw in tl.static_range(0, KW):
                w_in = w_idx + kw

                # base offset for this kernel position in weight, per oc
                kpos = (kt * KH + kh) * KW + kw  # in units of IC
                w_kpos_base = kpos * IC

                # base x offset (without IC)
                x_spatial_base = (
                    pid_n * stride_x_n
                    + d_in * stride_x_d
                    + h_in * stride_x_h
                    + w_in * stride_x_w
                )  # shape [BLOCK_N]

                for ick in range(0, IC_BLOCKS):
                    ic_off = ick * BLOCK_K + tl.arange(0, BLOCK_K)
                    ic_mask = ic_off < IC

                    # Load x tile: shape [BLOCK_N, BLOCK_K]
                    x_ptrs = x_ptr + x_spatial_base[:, None] + ic_off[None, :]
                    x_load_mask = s_mask[:, None] & ic_mask[None, :]
                    x_tile = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

                    # Load w tile: shape [BLOCK_M, BLOCK_K]
                    w_ptrs = w_ptr + oc_off[:, None] * stride_w_oc + (w_kpos_base + ic_off[None, :])
                    w_load_mask = oc_mask[:, None] & ic_mask[None, :]
                    w_tile = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

                    # acc += w_tile @ x_tile.T  -> (BLOCK_M, BLOCK_N)
                    acc += tl.dot(w_tile, tl.trans(x_tile))

    # Add bias+sum (already folded)
    bs = tl.load(b_ptr + oc_off, mask=oc_mask, other=0.0)  # [BLOCK_M]

    # bias (pre-sum) is the conv bias; sum_tensor is added AFTER leaky_relu in the original graph.
    # So we cannot fold them. Apply conv bias first, then leaky_relu, then sum_tensor, then clamp, then gelu.
    conv_bias = tl.load(b_ptr + oc_off, mask=oc_mask, other=0.0)
    sum_val = tl.load(s_ptr + oc_off, mask=oc_mask, other=0.0)

    # acc shape [BLOCK_M, BLOCK_N]
    y = acc + conv_bias[:, None]
    # leaky relu, slope=0.2
    y = tl.where(y >= 0.0, y, y * 0.2)
    # add sum_tensor (per OC)
    y = y + sum_val[:, None]
    # clamp
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    # GELU (exact)
    inv_sqrt2 = 0.7071067811865475
    y = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    # Store: output NDHWC layout
    # out offset = pid_n * stride_o_n + d * stride_o_d + h * stride_o_h + w * stride_o_w + oc
    out_spatial = (
        pid_n * stride_o_n
        + d_idx * stride_o_d
        + h_idx * stride_o_h
        + w_idx * stride_o_w
    )  # [BLOCK_N]
    out_ptrs = out_ptr + out_spatial[None, :] + oc_off[:, None]  # [BLOCK_M, BLOCK_N]
    store_mask = oc_mask[:, None] & s_mask[None, :]
    tl.store(out_ptrs, y, mask=store_mask)


def conv3d_fused(x, weight, bias, sum_tensor):
    """
    x: (N, IC, D_in, H_in, W_in)
    weight: (OC, IC, KT, KH, KW)
    bias: (OC,)
    sum_tensor: (OC, 1, 1, 1)

    Returns (N, OC, D_out, H_out, W_out)
    """
    N, IC, D_in, H_in, W_in = x.shape
    OC, _, KT, KH, KW = weight.shape
    D_out = D_in - KT + 1
    H_out = H_in - KH + 1
    W_out = W_in - KW + 1

    # Convert input to channels-last: NDHWC
    x_nhwc = x.permute(0, 2, 3, 4, 1).contiguous()  # (N, D_in, H_in, W_in, IC)
    # Convert weight to (OC, KT, KH, KW, IC)
    w_ndhwc = weight.permute(0, 2, 3, 4, 1).contiguous()

    # Allocate output in NDHWC
    out_nhwc = torch.empty((N, D_out, H_out, W_out, OC), device=x.device, dtype=x.dtype)

    s_flat = sum_tensor.contiguous().view(-1)
    b_flat = bias.contiguous()

    # strides in elements
    stride_x_n = D_in * H_in * W_in * IC
    stride_x_d = H_in * W_in * IC
    stride_x_h = W_in * IC
    stride_x_w = IC

    stride_w_oc = KT * KH * KW * IC

    stride_o_n = D_out * H_out * W_out * OC
    stride_o_d = H_out * W_out * OC
    stride_o_h = W_out * OC
    stride_o_w = OC

    DHW_out = D_out * H_out * W_out

    grid = lambda META: (
        N,
        triton.cdiv(OC, META['BLOCK_M']),
        triton.cdiv(DHW_out, META['BLOCK_N']),
    )

    conv3d_fused_kernel[grid](
        x_nhwc, w_ndhwc, b_flat, s_flat, out_nhwc,
        N, IC, OC,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        KT, KH, KW,
        stride_x_n, stride_x_d, stride_x_h, stride_x_w,
        stride_w_oc,
        stride_o_n, stride_o_d, stride_o_h, stride_o_w,
    )

    # Convert back to NCDHW
    out = out_nhwc.permute(0, 4, 1, 2, 3).contiguous()
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.kernel_size = kernel_size

    def forward(self, x):
        return conv3d_fused(x, self.conv.weight, self.conv.bias, self.sum_tensor)