import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 72}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 72}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 72}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64,  'BLOCK_K': 72}, num_warps=4, num_stages=2),
    ],
    key=['N_OUT', 'OC', 'K_TOTAL'],
)
@triton.jit
def conv_im2col_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    N_OUT, K_TOTAL: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oc = tl.arange(0, BLOCK_OC)
    mask_n = offs_n < N_OUT

    # decompose n -> (b, oh, ow)
    ow = offs_n % OW
    tmp = offs_n // OW
    oh = tmp % OH
    b = tmp // OH

    acc = tl.zeros((BLOCK_N, BLOCK_OC), dtype=tl.float32)

    # K_TOTAL = IC * KH * KW
    # Inner loop over K in tiles
    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_TOTAL

        # Decompose k -> (ic, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        ic_idx = tmp_k // KH

        # Gather x: shape [BLOCK_N, BLOCK_K]
        ih = oh[:, None] + kh_idx[None, :]
        iw = ow[:, None] + kw_idx[None, :]
        x_offs = (b[:, None] * (IC * IH * IW) +
                  ic_idx[None, :] * (IH * IW) +
                  ih * IW + iw)
        x_mask = mask_n[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

        # Load w: shape [BLOCK_K, BLOCK_OC]
        # w is (OC, IC, KH, KW) contiguous -> reshape to (OC, K_TOTAL)
        # w[oc, k] at offset oc * K_TOTAL + k
        w_offs = offs_k[:, None] * OC + offs_oc[None, :]
        # Actually, since reshape: w_flat[oc, k] = w[oc, ic, kh, kw], stride is (K_TOTAL, 1)
        # So w[oc, k] is at oc * K_TOTAL + k. For tile [K, OC] we want w[oc=offs_oc, k=offs_k]
        w_offs = offs_oc[None, :] * K_TOTAL + offs_k[:, None]
        w_mask = mask_k[:, None]
        w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile)

    # Bias + ReLU + HardSwish
    bias = tl.load(b_ptr + offs_oc)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)
    hs = (acc + 3.0) * (1.0 / 6.0)
    hs = tl.minimum(tl.maximum(hs, 0.0), 1.0)
    out = acc * hs

    # Store
    y_offs = (b[:, None] * (OC * OH * OW) +
              offs_oc[None, :] * (OH * OW) +
              oh[:, None] * OW + ow[:, None])
    mask = mask_n[:, None]
    tl.store(y_ptr + y_offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._w_flat = None

    def _get_w_flat(self):
        if self._w_flat is None or self._w_flat.device != self.conv.weight.device:
            w = self.conv.weight.contiguous()
            OC, IC, KH, KW = w.shape
            self._w_flat = w.view(OC, IC * KH * KW).contiguous()
        return self._w_flat

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        bias = self.conv.bias.contiguous().cuda()

        B, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        OH = IH - KH + 1
        OW = IW - KW + 1
        K_TOTAL = IC * KH * KW

        w_flat = w.view(OC, K_TOTAL).contiguous()

        y = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)

        N_OUT = B * OH * OW
        BLOCK_OC = OC  # 64, fits as single tile

        grid = lambda meta: (triton.cdiv(N_OUT, meta['BLOCK_N']),)

        conv_im2col_kernel[grid](
            x, w_flat, bias, y,
            B, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            N_OUT, K_TOTAL,
            BLOCK_OC=BLOCK_OC,
        )
        return y