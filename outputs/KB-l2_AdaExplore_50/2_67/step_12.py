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
    K_CONST: tl.constexpr,
    KH_CONST: tl.constexpr,
    KW_CONST: tl.constexpr,
):
    # grid: (N, ceil(OC/BLOCK_OC))
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # bias
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)

    npix = OH * OW
    acc_sum = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    inv_sqrt2 = 0.70710678118654752440

    # Loop over pixel tiles
    n_tiles = (npix + BLOCK_PIX - 1) // BLOCK_PIX
    for tile_idx in range(0, n_tiles):
        offs_p = tile_idx * BLOCK_PIX + tl.arange(0, BLOCK_PIX)
        mask_p = offs_p < npix
        oh = offs_p // OW
        ow = offs_p % OW

        acc = tl.zeros((BLOCK_OC, BLOCK_PIX), dtype=tl.float32)

        x_base = pid_n * IC * H * W
        for kidx in tl.static_range(0, K_CONST):
            ic = kidx // (KH_CONST * KW_CONST)
            rem = kidx % (KH_CONST * KW_CONST)
            kh = rem // KW_CONST
            kw = rem % KW_CONST
            ih = oh + kh
            iw = ow + kw
            x_off = x_base + ic * H * W + ih * W + iw
            x_val = tl.load(x_ptr + x_off, mask=mask_p, other=0.0)  # (BLOCK_PIX,)
            w_col = tl.load(w_ptr + offs_oc * K_CONST + kidx, mask=mask_oc, other=0.0)  # (BLOCK_OC,)
            acc += w_col[:, None] * x_val[None, :]

        acc += bias[:, None]
        gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
        valid = mask_oc[:, None] & mask_p[None, :]
        gelu = tl.where(valid, gelu, 0.0)
        acc_sum += tl.sum(gelu, axis=1)

    result = acc_sum * inv_npix
    out_off = pid_n * OC + offs_oc
    tl.store(out_ptr + out_off, result, mask=mask_oc)


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

        out = torch.empty((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_OC = 64
        BLOCK_PIX = 256
        K_CONST = IC * KH * KW

        grid = (N, triton.cdiv(OC, BLOCK_OC))

        conv_gelu_avgpool_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            inv_npix,
            BLOCK_OC=BLOCK_OC,
            BLOCK_PIX=BLOCK_PIX,
            K_CONST=K_CONST,
            KH_CONST=KH,
            KW_CONST=KW,
            num_warps=8,
            num_stages=2,
        )

        return out