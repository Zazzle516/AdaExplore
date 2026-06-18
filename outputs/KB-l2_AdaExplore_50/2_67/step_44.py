import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_gelu_avgpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW, OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    KSIZE_REAL: tl.constexpr,
    KSIZE_PAD: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    # grid: (N*OC, num_spatial_tiles)
    noc = tl.program_id(0)
    tile = tl.program_id(1)
    n = noc // OC
    oc = noc % OC

    OHOW = OH * OW
    inv = 1.0 / OHOW.to(tl.float32)

    # Load weights for this oc
    k_offs = tl.arange(0, KSIZE_PAD)
    k_mask = k_offs < KSIZE_REAL
    w_base = oc * KSIZE_REAL
    w_vals = tl.load(w_ptr + w_base + k_offs, mask=k_mask, other=0.0)

    bias = tl.load(b_ptr + oc)

    # decode k_offs into (ic, kh, kw) -> precompute kernel offset within input
    ic_idx = k_offs // (KH * KW)
    rem = k_offs % (KH * KW)
    kh_idx = rem // KW
    kw_idx = rem % KW
    # kernel offset relative to (oh=0, ow=0) for batch n:
    # offset = ic*IH*IW + kh*IW + kw
    k_off_in = ic_idx * (IH * IW) + kh_idx * IW + kw_idx  # (KSIZE_PAD,)

    n_base = n * IC * IH * IW

    sp_offs = tile * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    sp_mask = sp_offs < OHOW
    oh = sp_offs // OW
    ow = sp_offs % OW
    # base input offset per spatial position: oh*IW + ow
    sp_base = oh * IW + ow  # (BS,)

    x_idx = n_base + sp_base[:, None] + k_off_in[None, :]
    x_mask = sp_mask[:, None] & k_mask[None, :]
    x_vals = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

    prod = x_vals * w_vals[None, :]
    conv_out = tl.sum(prod, axis=1) + bias  # (BS,)

    # GELU (tanh approximation)
    c0 = 0.7978845608028654
    c1 = 0.044715
    x3 = conv_out * conv_out * conv_out
    inner = c0 * (conv_out + c1 * x3)
    t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    gelu = 0.5 * conv_out * (1.0 + t)

    gelu = tl.where(sp_mask, gelu, 0.0)
    partial = tl.sum(gelu, axis=0) * inv

    tl.atomic_add(out_ptr + n * OC + oc, partial)


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

        out = torch.zeros((N, OC), device=x.device, dtype=x.dtype)

        BLOCK_SPATIAL = 1024
        OHOW = OH * OW
        num_tiles = (OHOW + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL
        grid = (N * OC, num_tiles)
        KSIZE_REAL = IC * KH * KW
        # round up to next power of two
        KSIZE_PAD = 1
        while KSIZE_PAD < KSIZE_REAL:
            KSIZE_PAD *= 2
        conv_gelu_avgpool_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW, OC, OH, OW,
            KH, KW,
            IC,
            KSIZE_REAL,
            KSIZE_PAD,
            BLOCK_SPATIAL,
            num_warps=8,
            num_stages=3,
        )
        return out