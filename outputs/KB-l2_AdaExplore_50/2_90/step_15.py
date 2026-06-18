import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, sum_ptr, out_ptr,
    N, IC: tl.constexpr,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    NEG_SLOPE: tl.constexpr,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
):
    pid_n = tl.program_id(0)  # batch
    pid_m = tl.program_id(1)  # OC tile
    pid_s = tl.program_id(2)  # spatial tile

    OHW = OH * OW
    OSPATIAL = OD * OH * OW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial indices

    mask_m = offs_m < OC
    mask_s = offs_s < OSPATIAL

    # decompose spatial idx -> (od, oh, ow)
    od = offs_s // OHW
    rem = offs_s - od * OHW
    oh = rem // OW
    ow = rem - oh * OW

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # input base for this batch
    x_base = pid_n * IC * ID * IH * IW

    # Loop over IC, KD, KH, KW; full unroll via constexpr
    # weight layout: (OC, IC, KD, KH, KW)
    for ic in tl.static_range(0, IC):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw
                    # input indices
                    x_off = x_base + ic * (ID * IH * IW) + id_ * (IH * IW) + ih_ * IW + iw_
                    x_val = tl.load(x_ptr + x_off, mask=mask_s, other=0.0)  # [BLOCK_N]

                    # weight: [BLOCK_M] for each (ic,kd,kh,kw)
                    w_off = offs_m * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=mask_m, other=0.0)  # [BLOCK_M]

                    acc += w_val[:, None] * x_val[None, :]

    # add bias
    bias = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + bias[:, None]

    # leaky relu
    acc = tl.where(acc >= 0, acc, acc * NEG_SLOPE)

    # add sum_tensor (per-channel)
    s = tl.load(sum_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + s[:, None]

    # clamp
    acc = tl.maximum(acc, -1.0)
    acc = tl.minimum(acc, 1.0)

    # GELU exact
    inv_sqrt2 = 0.70710678118654752440
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Store: output layout (N, OC, OD, OH, OW)
    out_base = pid_n * OC * OSPATIAL
    out_off = out_base + offs_m[:, None] * OSPATIAL + offs_s[None, :]
    out_mask = mask_m[:, None] & mask_s[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv3d_fused(x, weight, bias, sum_tensor, neg_slope=0.2):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    sum_flat = sum_tensor.contiguous().view(-1)

    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    OSPATIAL = OD * OH * OW

    grid = (N, triton.cdiv(OC, BLOCK_M), triton.cdiv(OSPATIAL, BLOCK_N))

    conv3d_fused_kernel[grid](
        x, weight, bias, sum_flat, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        NEG_SLOPE=neg_slope,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        return conv3d_fused(
            x, self.conv.weight, self.conv.bias, self.sum_tensor, neg_slope=0.2
        )