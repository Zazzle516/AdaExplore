import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, sum_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    stride_xn, stride_xd, stride_xh, stride_xw, stride_xc,
    stride_on, stride_od, stride_oh, stride_ow, stride_oc,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
    BLOCK_K: tl.constexpr,  # IC tile
    IC_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_s = tl.program_id(1)  # spatial tile
    pid_m = tl.program_id(2)  # oc tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial indices

    # Decompose spatial idx -> (od, oh, ow)
    OHW = OH * OW
    od = offs_s // OHW
    rem = offs_s % OHW
    oh = rem // OW
    ow = rem % OW

    spatial_mask = offs_s < (OD * OH * OW)
    oc_mask = offs_m < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kernel positions kd, kh, kw and IC
    # weight layout: (OC, IC, KD, KH, KW) -> we'll use (OC, KD*KH*KW*IC) row-major
    # We choose weight stored as (OC, KD, KH, KW, IC) for coalesced K loads
    # K dim total = KD*KH*KW*IC

    # batch base pointer for x (NDHWC)
    x_n_off = pid_n * stride_xn

    for kd in tl.static_range(0, KD):
        id_ = od + kd  # input depth index
        for kh in tl.static_range(0, KH):
            ih = oh + kh
            for kw in tl.static_range(0, KW):
                iw = ow + kw
                # load x[n, id, ih, iw, :] for each spatial in tile (BLOCK_N, IC)
                x_base = x_n_off + id_ * stride_xd + ih * stride_xh + iw * stride_xw  # (BLOCK_N,)
                # weight base for (oc, kd, kh, kw, :): row offset
                # w layout: (OC, KD, KH, KW, IC), stride_w_oc = KD*KH*KW*IC
                w_kspatial = (kd * KH * KW + kh * KW + kw) * IC_CONST

                for ic_start in range(0, IC_CONST, BLOCK_K):
                    offs_k = ic_start + tl.arange(0, BLOCK_K)
                    k_mask = offs_k < IC_CONST

                    # x_tile: (BLOCK_N, BLOCK_K)
                    x_ptrs = x_ptr + x_base[:, None] + offs_k[None, :] * stride_xc
                    x_tile_mask = spatial_mask[:, None] & k_mask[None, :]
                    x_tile = tl.load(x_ptrs, mask=x_tile_mask, other=0.0)

                    # w_tile: (BLOCK_M, BLOCK_K)
                    # w[oc, kd, kh, kw, ic] = w_ptr + oc * (KD*KH*KW*IC) + w_kspatial + ic
                    w_ptrs = w_ptr + offs_m[:, None] * (KD * KH * KW * IC_CONST) + (w_kspatial + offs_k[None, :])
                    w_tile_mask = oc_mask[:, None] & k_mask[None, :]
                    w_tile = tl.load(w_ptrs, mask=w_tile_mask, other=0.0)

                    # acc += w @ x.T  -> (BLOCK_M, BLOCK_N)
                    acc += tl.dot(w_tile, tl.trans(x_tile))

    # add bias
    bias = tl.load(b_ptr + offs_m, mask=oc_mask, other=0.0)  # (BLOCK_M,)
    acc += bias[:, None]

    # leaky relu
    acc = tl.where(acc >= 0, acc, 0.2 * acc)

    # add sum_tensor (per OC scalar)
    s = tl.load(sum_ptr + offs_m, mask=oc_mask, other=0.0)  # (BLOCK_M,)
    acc += s[:, None]

    # clamp
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # gelu exact
    inv_sqrt2 = 0.70710678118654752440
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # store: out layout NDHWC -> out[n, od, oh, ow, oc]
    # offsets: pid_n*stride_on + od*stride_od + oh*stride_oh + ow*stride_ow + oc*stride_oc
    out_spatial = od * stride_od + oh * stride_oh + ow * stride_ow  # (BLOCK_N,)
    out_ptrs = out_ptr + pid_n * stride_on + out_spatial[None, :] + offs_m[:, None] * stride_oc
    out_mask = oc_mask[:, None] & spatial_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # Weight shape: (OC, IC, KD, KH, KW). Reorder to (OC, KD, KH, KW, IC) and store contiguous.
        w = conv.weight.detach().clone()  # (OC, IC, KD, KH, KW)
        w = w.permute(0, 2, 3, 4, 1).contiguous()  # (OC, KD, KH, KW, IC)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(conv.bias.detach().clone())
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # x: (N, IC, ID, IH, IW) -> NDHWC
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        # Convert to NDHWC
        x_nhwc = x.permute(0, 2, 3, 4, 1).contiguous()  # (N, ID, IH, IW, IC)

        # Output in NDHWC
        out_nhwc = torch.empty((N, OD, OH, OW, OC), device=x.device, dtype=x.dtype)

        sum_flat = self.sum_tensor.contiguous().view(-1)

        # strides in elements
        stride_xn = ID * IH * IW * IC
        stride_xd = IH * IW * IC
        stride_xh = IW * IC
        stride_xw = IC
        stride_xc = 1

        stride_on = OD * OH * OW * OC
        stride_od = OH * OW * OC
        stride_oh = OW * OC
        stride_ow = OC
        stride_oc = 1

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 8 if IC <= 8 else 16

        spatial_total = OD * OH * OW
        grid = (N, (spatial_total + BLOCK_N - 1) // BLOCK_N, (OC + BLOCK_M - 1) // BLOCK_M)

        conv3d_fused_kernel[grid](
            x_nhwc, self.weight, self.bias, sum_flat, out_nhwc,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            stride_xn, stride_xd, stride_xh, stride_xw, stride_xc,
            stride_on, stride_od, stride_oh, stride_ow, stride_oc,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            IC_CONST=IC,
            num_warps=4, num_stages=3,
        )

        # Convert back to NCDHW
        out = out_nhwc.permute(0, 4, 1, 2, 3).contiguous()
        return out