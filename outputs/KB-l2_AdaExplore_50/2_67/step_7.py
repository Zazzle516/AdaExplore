import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PIX': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PIX': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PIX': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PIX': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PIX': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PIX': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PIX': 128}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'OH', 'OW', 'K_CONST'],
)
@triton.jit
def conv_gelu_avgpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    inv_npix,
    K_CONST: tl.constexpr,
    K_PAD: tl.constexpr,
    KH_CONST: tl.constexpr,
    KW_CONST: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_PIX: tl.constexpr,
):
    # grid: (N, ceil(OC/BLOCK_OC))
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # bias
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)

    # Load weights once: shape (K_PAD, BLOCK_OC) for tl.dot as RHS
    offs_k = tl.arange(0, K_PAD)
    mask_k = offs_k < K_CONST
    w_ptrs = w_ptr + offs_oc[None, :] * K_CONST + offs_k[:, None]
    w_tile = tl.load(w_ptrs, mask=mask_oc[None, :] & mask_k[:, None], other=0.0)  # (K_PAD, BLOCK_OC)

    # Decompose K index into (ic, kh, kw)
    KHKW = KH_CONST * KW_CONST
    ic_k = offs_k // KHKW
    rem_k = offs_k % KHKW
    kh_k = rem_k // KW_CONST
    kw_k = rem_k % KW_CONST

    HW = H * W
    x_base = pid_n * IC * HW

    npix = OH * OW
    acc_sum = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    inv_sqrt2 = 0.70710678118654752440

    n_tiles = (npix + BLOCK_PIX - 1) // BLOCK_PIX
    for tile_idx in range(0, n_tiles):
        offs_p = tile_idx * BLOCK_PIX + tl.arange(0, BLOCK_PIX)
        mask_p = offs_p < npix
        oh = offs_p // OW
        ow = offs_p % OW

        # Build im2col x_tile: shape (BLOCK_PIX, K_PAD)
        # addr = x_base + ic*HW + (oh+kh)*W + (ow+kw)
        addr = (x_base
                + ic_k[None, :] * HW
                + (oh[:, None] + kh_k[None, :]) * W
                + (ow[:, None] + kw_k[None, :]))
        x_tile = tl.load(x_ptr + addr,
                         mask=mask_p[:, None] & mask_k[None, :],
                         other=0.0)  # (BLOCK_PIX, K_PAD)

        # GEMM: (BLOCK_PIX, K_PAD) @ (K_PAD, BLOCK_OC) -> (BLOCK_PIX, BLOCK_OC)
        acc = tl.dot(x_tile, w_tile, out_dtype=tl.float32)

        acc += bias[None, :]
        gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
        valid = mask_p[:, None] & mask_oc[None, :]
        gelu = tl.where(valid, gelu, 0.0)
        acc_sum += tl.sum(gelu, axis=0)

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

        K_CONST = IC * KH * KW
        # next power of two >= K_CONST, min 16 for tl.dot
        K_PAD = 1
        while K_PAD < max(K_CONST, 16):
            K_PAD *= 2

        grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']))

        conv_gelu_avgpool_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            inv_npix,
            K_CONST=K_CONST,
            K_PAD=K_PAD,
            KH_CONST=KH,
            KW_CONST=KW,
        )

        return out