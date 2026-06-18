import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_gelu_avgpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    inv_npix,
    BLOCK_OC: tl.constexpr,
    BLOCK_PIX: tl.constexpr,
    IC_CONST: tl.constexpr,
    KH_CONST: tl.constexpr,
    KW_CONST: tl.constexpr,
):
    # grid: (N, ceil(OC/BLOCK_OC), ceil(OH*OW/BLOCK_PIX))
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_p = pid_p * BLOCK_PIX + tl.arange(0, BLOCK_PIX)

    mask_oc = offs_oc < OC
    mask_p = offs_p < (OH * OW)

    oh = offs_p // OW
    ow = offs_p % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_PIX), dtype=tl.float32)

    # x: (N, IC, H, W), w: (OC, IC, KH, KW)
    for ic in tl.static_range(0, IC_CONST):
        for kh in tl.static_range(0, KH_CONST):
            for kw in tl.static_range(0, KW_CONST):
                ih = oh + kh
                iw = ow + kw
                x_off = pid_n * IC * H * W + ic * H * W + ih * W + iw
                x_val = tl.load(x_ptr + x_off, mask=mask_p, other=0.0)  # (BLOCK_PIX,)

                w_off = offs_oc * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # (BLOCK_OC,)

                acc += w_val[:, None] * x_val[None, :]

    # add bias
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += bias[:, None]

    # GELU (erf-based exact)
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # mask out invalid pixels
    valid = mask_oc[:, None] & mask_p[None, :]
    gelu = tl.where(valid, gelu, 0.0)

    # partial sum over pixels in this block
    partial = tl.sum(gelu, axis=1) * inv_npix  # (BLOCK_OC,)

    # atomic add to output (N, OC)
    out_off = pid_n * OC + offs_oc
    tl.atomic_add(out_ptr + out_off, partial, mask=mask_oc)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()

        N, IC, H, W = x.shape
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        npix = OH * OW
        inv_npix = 1.0 / npix

        out = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_OC = 32
        BLOCK_PIX = 128

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(npix, BLOCK_PIX))

        conv_gelu_avgpool_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            inv_npix,
            BLOCK_OC=BLOCK_OC,
            BLOCK_PIX=BLOCK_PIX,
            IC_CONST=IC,
            KH_CONST=KH,
            KW_CONST=KW,
            num_warps=4,
            num_stages=2,
        )

        return out