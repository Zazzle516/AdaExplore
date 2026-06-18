import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
    ],
    key=['N', 'OC', 'OD', 'OH', 'OW', 'IC', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, sum_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    M,  # = OD * OH * OW
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IC_CONST: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial output indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC indices

    mask_m = offs_m < M
    mask_n = offs_n < OC

    # decompose offs_m into od, oh, ow
    OHW = OH * OW
    od = offs_m // OHW
    rem = offs_m - od * OHW if False else (offs_m % OHW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = IC * KD * KH * KW; fully unrolled
    for ic in tl.static_range(0, IC_CONST):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    # input offset per m: pid_b*IC*ID*IH*IW + ic*ID*IH*IW + (od+kd)*IH*IW + (oh+kh)*IW + (ow+kw)
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw
                    x_off = (pid_b * IC * ID * IH * IW
                             + ic * ID * IH * IW
                             + id_ * IH * IW
                             + ih_ * IW
                             + iw_)
                    x_vec = tl.load(x_ptr + x_off, mask=mask_m, other=0.0)  # [BLOCK_M]

                    # weight: w[oc, ic, kd, kh, kw]; layout (OC, IC, KD, KH, KW)
                    w_off = (offs_n * (IC * KD * KH * KW)
                             + ic * (KD * KH * KW)
                             + kd * (KH * KW)
                             + kh * KW
                             + kw)
                    w_vec = tl.load(w_ptr + w_off, mask=mask_n, other=0.0)  # [BLOCK_N]

                    acc += x_vec[:, None] * w_vec[None, :]

    # Load bias and sum tensor (both per-OC)
    b_vec = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    s_vec = tl.load(sum_ptr + offs_n, mask=mask_n, other=0.0)

    acc = acc + b_vec[None, :]
    # LeakyReLU(0.2)
    acc = tl.where(acc >= 0, acc, acc * 0.2)
    # add sum_tensor
    acc = acc + s_vec[None, :]
    # clamp
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)
    # GELU (exact via erf)
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # store: out shape (N, OC, OD, OH, OW)
    # out offset = pid_b*OC*M + offs_n*M + offs_m
    out_off = pid_b * OC * M + offs_n[None, :] * M + offs_m[:, None]
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


def conv3d_fused(x, w, b, sum_tensor):
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    sum_flat = sum_tensor.view(-1).contiguous()

    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = w.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    M = OD * OH * OW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
        N,
    )

    conv3d_fused_kernel[grid](
        x, w, b, sum_flat, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        M,
        IC_CONST=IC,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        return conv3d_fused(x, self.conv.weight, self.conv.bias, self.sum_tensor)