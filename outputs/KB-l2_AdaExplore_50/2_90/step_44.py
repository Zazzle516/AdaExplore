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
    BLOCK_M: tl.constexpr,  # output spatial tile
    BLOCK_N: tl.constexpr,  # OC tile
    BLOCK_K: tl.constexpr,  # IC tile
):
    pid_m = tl.program_id(0)  # spatial tile
    pid_n = tl.program_id(1)  # OC tile
    pid_b = tl.program_id(2)  # batch

    OSP = OD * OH * OW
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output spatial positions
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channels

    mask_m = offs_m < OSP
    mask_n = offs_n < OC

    # Decompose offs_m -> (od, oh, ow)
    od = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kernel volume and IC
    KVOL = KD * KH * KW

    # Pointers base for this batch input
    x_batch_ptr = x_ptr + pid_b * IC * ID * IH * IW

    for kk in range(0, KVOL):
        kd = kk // (KH * KW)
        khw = kk % (KH * KW)
        kh = khw // KW
        kw = khw % KW

        id_ = od + kd
        ih_ = oh + kh
        iw_ = ow + kw
        # No padding so all valid (since output sized accordingly)
        # spatial offset into input plane
        in_spatial_off = id_ * (IH * IW) + ih_ * IW + iw_  # [BLOCK_M]

        for ic_start in range(0, IC, BLOCK_K):
            offs_k = ic_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < IC

            # Load x: [BLOCK_M, BLOCK_K]
            x_offs = in_spatial_off[:, None] + offs_k[None, :] * (ID * IH * IW)
            x_mask = mask_m[:, None] & mask_k[None, :]
            x_vals = tl.load(x_batch_ptr + x_offs, mask=x_mask, other=0.0)

            # Load w: [BLOCK_K, BLOCK_N]
            # w shape: (OC, IC, KD, KH, KW), stride in last dim = 1
            # for fixed kd,kh,kw, ic varies along IC
            w_offs = (offs_n[None, :] * (IC * KVOL)
                      + offs_k[:, None] * KVOL
                      + kk)
            w_mask = mask_k[:, None] & mask_n[None, :]
            w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

            acc += tl.dot(x_vals, w_vals)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # LeakyReLU(0.2)
    acc = tl.where(acc >= 0, acc, acc * 0.2)

    # Add sum_tensor (per OC)
    s = tl.load(sum_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + s[None, :]

    # Clamp
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # GELU exact
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Store: output layout (N, OC, OD, OH, OW)
    out_batch_ptr = out_ptr + pid_b * OC * OSP
    out_offs = offs_n[None, :] * OSP + offs_m[:, None]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_batch_ptr + out_offs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()
        sum_flat = self.sum_tensor.view(-1).contiguous()

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 16  # IC=8, padded

        OSP = OD * OH * OW
        grid = (
            (OSP + BLOCK_M - 1) // BLOCK_M,
            (OC + BLOCK_N - 1) // BLOCK_N,
            N,
        )

        conv3d_fused_kernel[grid](
            x, weight, bias, sum_flat, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        return out