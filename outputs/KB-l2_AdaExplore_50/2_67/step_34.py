import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SPATIAL': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SPATIAL': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SPATIAL': 2048}, num_warps=8, num_stages=3),
    ],
    key=['N', 'OC', 'OH', 'OW'],
)
@triton.jit
def conv_gelu_avgpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC,
    OH: tl.constexpr, OW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    KSIZE: tl.constexpr,
    KSIZE_P2: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    # grid: (N, OC)
    n = tl.program_id(0)
    oc = tl.program_id(1)

    OHOW: tl.constexpr = OH * OW
    inv: tl.constexpr = 1.0 / OHOW

    # Load weights for this oc: shape (IC, KH, KW) -> flatten to IC*KH*KW
    k_offs = tl.arange(0, KSIZE_P2)
    k_mask = k_offs < KSIZE
    w_base = oc * IC_C * KH * KW
    w_vals = tl.load(w_ptr + w_base + k_offs, mask=k_mask, other=0.0)  # (KSIZE_P2,)

    bias = tl.load(b_ptr + oc)

    acc = tl.zeros((), dtype=tl.float32)

    # decode k_offs into (ic, kh, kw); use safe modulo for padded entries (masked out)
    k_safe = tl.where(k_mask, k_offs, 0)
    ic_idx = k_safe // (KH * KW)
    rem = k_safe % (KH * KW)
    kh_idx = rem // KW
    kw_idx = rem % KW

    # precompute kernel offsets: k_off[k] = ic*IH*IW + kh*IW + kw
    k_off = ic_idx * (IH * IW) + kh_idx * IW + kw_idx  # (KSIZE_P2,)
    base_n = n * IC * IH * IW

    num_blocks: tl.constexpr = (OHOW + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL

    for blk in range(0, num_blocks):
        sp_offs = blk * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
        sp_mask = sp_offs < OHOW
        oh = sp_offs // OW
        ow = sp_offs % OW

        # base offset per spatial position: oh*IW + ow
        sp_base = oh * IW + ow  # (BS,)

        x_idx = base_n + sp_base[:, None] + k_off[None, :]
        x_mask = sp_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

        # multiply by weights and sum over ksize
        prod = x_vals * w_vals[None, :]
        conv_out = tl.sum(prod, axis=1) + bias  # (BS,)

        # GELU (tanh approximation)
        c0 = 0.7978845608028654  # sqrt(2/pi)
        c1 = 0.044715
        x3 = conv_out * conv_out * conv_out
        inner = c0 * (conv_out + c1 * x3)
        t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        gelu = 0.5 * conv_out * (1.0 + t)

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
        # next power of two >= KSIZE
        KSIZE_P2 = 1
        while KSIZE_P2 < KSIZE:
            KSIZE_P2 *= 2
        grid = (N, OC)
        conv_gelu_avgpool_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC,
            OH, OW,
            KH, KW,
            IC,
            KSIZE,
            KSIZE_P2,
        )
        return out