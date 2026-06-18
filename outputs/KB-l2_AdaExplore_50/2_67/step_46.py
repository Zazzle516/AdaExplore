import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SPATIAL': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 1024}, num_warps=8, num_stages=2),
    ],
    key=['N', 'IC', 'IH', 'IW', 'OC'],
)
@triton.jit
def conv_gelu_avgpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    inv,
    N, IC, IH, IW,
    OC,
    OH: tl.constexpr, OW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    KSIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    # grid: (N, OC)
    n = tl.program_id(0)
    oc = tl.program_id(1)

    OHOW: tl.constexpr = OH * OW

    # Load weights for this oc: shape (IC, KH, KW) -> flatten to IC*KH*KW
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < KSIZE
    w_base = oc * IC_C * KH * KW
    w_vals = tl.load(w_ptr + w_base + k_offs, mask=k_mask, other=0.0)  # (BLOCK_K,)

    bias = tl.load(b_ptr + oc)

    acc = tl.zeros((), dtype=tl.float32)

    # decode k_offs into (ic, kh, kw)
    ic_idx = k_offs // (KH * KW)
    rem = k_offs % (KH * KW)
    kh_idx = rem // KW
    kw_idx = rem % KW

    num_blocks = (OHOW + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL

    for blk in range(0, num_blocks):
        sp_offs = blk * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
        sp_mask = sp_offs < OHOW
        oh = sp_offs // OW
        ow = sp_offs % OW

        # For each spatial position, compute conv output
        # out[sp] = sum over ksize of x[n, ic, oh+kh, ow+kw] * w[ksize]
        # Build input indices: shape (BLOCK_SPATIAL, BLOCK_K)
        ih = oh[:, None] + kh_idx[None, :]  # (BS, BLOCK_K)
        iw = ow[:, None] + kw_idx[None, :]
        ic_b = ic_idx[None, :].broadcast_to(BLOCK_SPATIAL, BLOCK_K)

        x_idx = ((n * IC + ic_b) * IH + ih) * IW + iw
        x_mask = sp_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

        # multiply by weights and sum over ksize
        prod = x_vals * w_vals[None, :]
        conv_out = tl.sum(prod, axis=1) + bias  # (BS,)

        # GELU (tanh approximation)
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        c0 = 0.7978845608028654  # sqrt(2/pi)
        c1 = 0.044715
        x3 = conv_out * conv_out * conv_out
        inner = c0 * (conv_out + c1 * x3)
        # tanh via sigmoid: tanh(z) = 2*sigmoid(2z) - 1
        t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        gelu = 0.5 * conv_out * (1.0 + t)

        # mask out invalid positions
        gelu = tl.where(sp_mask, gelu, 0.0)
        acc += tl.sum(gelu, axis=0)

    result = acc * inv
    tl.store(out_ptr + n * OC + oc, result)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC), device=x.device, dtype=x.dtype)

        KSIZE = IC * KH * KW
        # next power of 2 for BLOCK_K
        BLOCK_K = 1
        while BLOCK_K < KSIZE:
            BLOCK_K *= 2
        inv = 1.0 / float(OH * OW)
        grid = (N, OC)
        conv_gelu_avgpool_kernel[grid](
            x, w, b, out,
            inv,
            N, IC, IH, IW,
            OC,
            OH, OW,
            KH, KW,
            IC,
            KSIZE,
            BLOCK_K,
        )
        return out