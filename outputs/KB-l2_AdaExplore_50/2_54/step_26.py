import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_fused_kernel(
    x_ptr, w_ptr, b_ptr, mult_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # weight shape: (OC, IC, KH, KW)
    # x shape: (N, IC, IH, IW)
    for ic in range(0, IC_C):
        for kh in range(0, KH):
            for kw in range(0, KW):
                ih = oh + kh
                iw = ow + kw
                # Load x: (BLOCK_SP,)
                x_idx = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                x_vals = tl.load(x_ptr + x_idx, mask=sp_mask, other=0.0)  # (BLOCK_SP,)
                # Load w: (BLOCK_OC,)
                w_idx = oc_offs * (IC * KH * KW) + ic * KH * KW + kh * KW + kw
                w_vals = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)  # (BLOCK_OC,)
                acc += w_vals[:, None] * x_vals[None, :]

    # Add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    # Multiply by multiplier (per-channel)
    mult = tl.load(mult_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc * mult[:, None]

    # LeakyReLU (negative_slope=0.01)
    acc = tl.where(acc >= 0, acc, acc * 0.01)

    # GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Store
    out_idx = pid_n * OC * OH * OW + oc_offs[:, None] * (OH * OW) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        mult = self.multiplier.contiguous().view(-1)

        N, IC, IH, IW = x.shape
        OC = w.shape[0]
        KH = w.shape[2]
        KW = w.shape[3]
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 128

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

        conv_fused_kernel[grid](
            x, w, b, mult, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH=KH, KW=KW,
            IC_C=IC,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )
        return out