import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'IC', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr,       # [N, IC, ID, IH, IW]
    w_ptr,       # [OC, IC, KD, KH, KW]
    b_ptr,       # [OC]
    sum_ptr,     # [OC]
    out_ptr,     # [N, OC, OD, OH, OW]
    N, IC,
    ID, IH, IW,
    OC,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
    BLOCK_K: tl.constexpr,  # K-dim tile (over IC)
):
    pid_n = tl.program_id(0)         # batch index
    pid_m = tl.program_id(1)         # OC tile
    pid_s = tl.program_id(2)         # spatial tile

    spatial = OD * OH * OW
    # offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # OC indices
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)   # spatial indices

    mask_m = offs_m < OC
    mask_s = offs_s < spatial

    # Decompose spatial idx -> (od, oh, ow)
    ow = offs_s % OW
    tmp = offs_s // OW
    oh = tmp % OH
    od = tmp // OH

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over KD, KH, KW, IC
    # weight layout: w[oc, ic, kd, kh, kw], strides = (IC*KD*KH*KW, KD*KH*KW, KH*KW, KW, 1)
    # input layout : x[n, ic, id, ih, iw]
    KDHW = KD * KH * KW
    IC_KDHW = IC * KDHW

    for kd in tl.static_range(0, KD):
        id_ = od + kd  # input depth index
        for kh in tl.static_range(0, KH):
            ih_ = oh + kh
            for kw in tl.static_range(0, KW):
                iw_ = ow + kw
                # input base offset for n, spatial position
                # x[n, ic, id_, ih_, iw_]
                # for each ic in K-loop
                spatial_in_off = id_ * (IH * IW) + ih_ * IW + iw_  # [BLOCK_N]
                n_off = pid_n * (IC * ID * IH * IW)
                # weight base: w[oc, ic, kd, kh, kw]
                w_kpos = kd * (KH * KW) + kh * KW + kw

                for ic_start in range(0, IC, BLOCK_K):
                    ic_idx = ic_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]
                    mask_k = ic_idx < IC

                    # Load x_tile: [BLOCK_K, BLOCK_N]
                    x_off = n_off + ic_idx[:, None] * (ID * IH * IW) + spatial_in_off[None, :]
                    x_mask = mask_k[:, None] & mask_s[None, :]
                    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                    # Load w_tile: [BLOCK_M, BLOCK_K]
                    w_off = offs_m[:, None] * IC_KDHW + ic_idx[None, :] * KDHW + w_kpos
                    w_mask = mask_m[:, None] & mask_k[None, :]
                    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                    acc += tl.dot(w_tile, x_tile)

    # Add conv bias
    bias = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + bias[:, None]

    # LeakyReLU(0.2)
    acc = tl.where(acc > 0, acc, acc * 0.2)

    # Add sum_tensor (per OC)
    s = tl.load(sum_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + s[:, None]

    # Clamp
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # GELU exact
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Store: out[n, oc, od, oh, ow]
    out_off = pid_n * (OC * spatial) + offs_m[:, None] * spatial + offs_s[None, :]
    out_mask = mask_m[:, None] & mask_s[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        s = self.sum_tensor.contiguous().view(-1)

        N, IC, ID, IH, IW = x.shape
        OC, _, KD, KH, KW = w.shape
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)
        spatial = OD * OH * OW

        grid = lambda META: (
            N,
            triton.cdiv(OC, META['BLOCK_M']),
            triton.cdiv(spatial, META['BLOCK_N']),
        )

        conv3d_fused_kernel[grid](
            x, w, b, s, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
        )
        return out