import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_OH': 2, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_OH': 2, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_OH': 1, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_OH': 2, 'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_OH': 2, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_OH': 1, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OD', 'OH', 'OW', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, sum_ptr, out_ptr,
    N, IC: tl.constexpr,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    NEG_SLOPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_n = tl.program_id(0)        # batch
    pid_m = tl.program_id(1)        # OC tile
    pid_s = tl.program_id(2)        # spatial tile over (od, oh_tile, ow_tile)

    OW_TILES = tl.cdiv(OW, BLOCK_OW)
    OH_TILES = tl.cdiv(OH, BLOCK_OH)
    TILES_PER_OD = OH_TILES * OW_TILES

    od = pid_s // TILES_PER_OD
    rem = pid_s - od * TILES_PER_OD
    oh_tile = rem // OW_TILES
    ow_tile = rem - oh_tile * OW_TILES

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_oh = oh_tile * BLOCK_OH + tl.arange(0, BLOCK_OH)
    offs_ow = ow_tile * BLOCK_OW + tl.arange(0, BLOCK_OW)

    mask_m = offs_m < OC
    mask_oh = offs_oh < OH
    mask_ow = offs_ow < OW
    mask_sp = mask_oh[:, None] & mask_ow[None, :]  # [BLOCK_OH, BLOCK_OW]

    # accumulator [BLOCK_M, BLOCK_OH, BLOCK_OW]
    acc = tl.zeros((BLOCK_M, BLOCK_OH, BLOCK_OW), dtype=tl.float32)

    x_base = pid_n * IC * ID * IH * IW

    for ic in tl.static_range(0, IC):
        for kd in tl.static_range(0, KD):
            id_ = od + kd
            for kh in tl.static_range(0, KH):
                ih_ = offs_oh + kh  # [BLOCK_OH]
                for kw in tl.static_range(0, KW):
                    iw_ = offs_ow + kw  # [BLOCK_OW]
                    x_off = (x_base
                             + ic * (ID * IH * IW)
                             + id_ * (IH * IW)
                             + ih_[:, None] * IW
                             + iw_[None, :])
                    x_val = tl.load(x_ptr + x_off, mask=mask_sp, other=0.0)  # [BLOCK_OH, BLOCK_OW]

                    w_off = (offs_m * (IC * KD * KH * KW)
                             + ic * (KD * KH * KW)
                             + kd * (KH * KW)
                             + kh * KW + kw)
                    w_val = tl.load(w_ptr + w_off, mask=mask_m, other=0.0)  # [BLOCK_M]

                    acc += w_val[:, None, None] * x_val[None, :, :]

    bias = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + bias[:, None, None]
    acc = tl.where(acc >= 0, acc, acc * NEG_SLOPE)
    s = tl.load(sum_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + s[:, None, None]
    acc = tl.maximum(acc, -1.0)
    acc = tl.minimum(acc, 1.0)
    inv_sqrt2 = 0.70710678118654752440
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    OSPATIAL = OD * OH * OW
    OHW = OH * OW
    out_base = pid_n * OC * OSPATIAL
    out_off = (out_base
               + offs_m[:, None, None] * OSPATIAL
               + od * OHW
               + offs_oh[None, :, None] * OW
               + offs_ow[None, None, :])
    out_mask = mask_m[:, None, None] & mask_sp[None, :, :]
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

    def grid(meta):
        return (
            N,
            triton.cdiv(OC, meta['BLOCK_M']),
            OD * triton.cdiv(OH, meta['BLOCK_OH']) * triton.cdiv(OW, meta['BLOCK_OW']),
        )

    conv3d_fused_kernel[grid](
        x, weight, bias, sum_flat, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        NEG_SLOPE=neg_slope,
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