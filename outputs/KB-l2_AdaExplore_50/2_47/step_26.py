import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OUT_SPATIAL', 'IC', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv3d_mish_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    OUT_SPATIAL,
    # strides for x in NCDHW (in elements)
    x_sN, x_sC, x_sD, x_sH, x_sW,
    # strides for w in (OC, IC, KD, KH, KW)
    w_sOC, w_sIC, w_sKD, w_sKH, w_sKW,
    # strides for out NCDHW
    o_sN, o_sC, o_sD, o_sH, o_sW,
    BLOCK_M: tl.constexpr,   # OC tile
    BLOCK_N: tl.constexpr,   # spatial tile
    BLOCK_K: tl.constexpr,   # IC tile
):
    pid_n = tl.program_id(0)             # batch index
    pid_m = tl.program_id(1)             # OC tile
    pid_s = tl.program_id(2)             # spatial tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)         # OC
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)         # spatial linear idx in output

    mask_m = offs_m < OC
    mask_s = offs_s < OUT_SPATIAL

    # decompose spatial idx -> (od, oh, ow)
    ow = offs_s % OW
    tmp = offs_s // OW
    oh = tmp % OH
    od = tmp // OH

    # base input coordinate for each output position
    id_base = od * SD - PD     # [BLOCK_N]
    ih_base = oh * SH - PH
    iw_base = ow * SW - PW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kernel positions and IC
    for kd in tl.static_range(0, KD):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                id_ = id_base + kd
                ih_ = ih_base + kh
                iw_ = iw_base + kw
                in_bounds = (id_ >= 0) & (id_ < ID) & (ih_ >= 0) & (ih_ < IH) & (iw_ >= 0) & (iw_ < IW)

                # input offset per spatial position (without channel)
                x_spatial_off = (pid_n * x_sN
                                 + id_ * x_sD
                                 + ih_ * x_sH
                                 + iw_ * x_sW)  # [BLOCK_N]

                # weight offset per OC for this (kd,kh,kw), still need IC
                w_kpos_off = (offs_m * w_sOC
                              + kd * w_sKD
                              + kh * w_sKH
                              + kw * w_sKW)  # [BLOCK_M]

                # Loop over IC in BLOCK_K chunks
                for ic_start in range(0, IC, BLOCK_K):
                    offs_k = ic_start + tl.arange(0, BLOCK_K)
                    mask_k = offs_k < IC

                    # x: [BLOCK_K, BLOCK_N]
                    x_ptrs = x_ptr + offs_k[:, None] * x_sC + x_spatial_off[None, :]
                    x_mask = mask_k[:, None] & in_bounds[None, :] & mask_s[None, :]
                    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                    # w: [BLOCK_M, BLOCK_K]
                    w_ptrs = w_ptr + w_kpos_off[:, None] + offs_k[None, :] * w_sIC
                    w_mask = mask_m[:, None] & mask_k[None, :]
                    w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                    acc += tl.dot(w_vals, x_vals, allow_tf32=True)

    # add bias
    b_vals = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc += b_vals[:, None]

    # fused mish + tanh
    # mish(x) = x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    t1 = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    m = acc * t1
    t2 = 2.0 * tl.sigmoid(2.0 * m) - 1.0

    # output offsets: [BLOCK_M, BLOCK_N]
    out_off = (pid_n * o_sN
               + offs_m[:, None] * o_sC
               + od[None, :] * o_sD
               + oh[None, :] * o_sH
               + ow[None, :] * o_sW)
    out_mask = mask_m[:, None] & mask_s[None, :]
    tl.store(out_ptr + out_off, t2, mask=out_mask)


def conv3d_mish_tanh(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    SD, SH, SW = stride
    PD, PH, PW = padding
    OD = (ID + 2 * PD - KD) // SD + 1
    OH = (IH + 2 * PH - KH) // SH + 1
    OW = (IW + 2 * PW - KW) // SW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    OUT_SPATIAL = OD * OH * OW

    grid = lambda META: (
        N,
        triton.cdiv(OC, META['BLOCK_M']),
        triton.cdiv(OUT_SPATIAL, META['BLOCK_N']),
    )

    conv3d_mish_tanh_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        OUT_SPATIAL,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = tuple(stride)
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = tuple(padding)

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight
        b = self.conv.bias
        return conv3d_mish_tanh(x, w, b, self.stride, self.padding)