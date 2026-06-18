import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_PIX': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_PIX': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_PIX': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_PIX': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_PIX': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_PIX': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_PIX': 256}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'OH', 'OW', 'K_CONST'],
)
@triton.jit
def conv_gelu_avgpool_partial_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    N, IC, H, W,
    OC, OH, OW,
    NPIX_TILES,
    K_CONST: tl.constexpr,
    K_PAD: tl.constexpr,
    KH_CONST: tl.constexpr,
    KW_CONST: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_PIX: tl.constexpr,
):
    # grid: (N, NPIX_TILES)
    pid_n = tl.program_id(0)
    pid_pt = tl.program_id(1)

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)

    offs_k = tl.arange(0, K_PAD)
    mask_k = offs_k < K_CONST
    w_ptrs = w_ptr + offs_oc[None, :] * K_CONST + offs_k[:, None]
    w_tile = tl.load(w_ptrs, mask=mask_oc[None, :] & mask_k[:, None], other=0.0)

    KHKW = KH_CONST * KW_CONST
    ic_k = offs_k // KHKW
    rem_k = offs_k % KHKW
    kh_k = rem_k // KW_CONST
    kw_k = rem_k % KW_CONST

    HW = H * W
    x_base = pid_n * IC * HW

    npix = OH * OW
    inv_sqrt2 = 0.70710678118654752440

    offs_p = pid_pt * BLOCK_PIX + tl.arange(0, BLOCK_PIX)
    mask_p = offs_p < npix
    oh = offs_p // OW
    ow = offs_p % OW

    addr = (x_base
            + ic_k[None, :] * HW
            + (oh[:, None] + kh_k[None, :]) * W
            + (ow[:, None] + kw_k[None, :]))
    x_tile = tl.load(x_ptr + addr,
                     mask=mask_p[:, None] & mask_k[None, :],
                     other=0.0)

    acc = tl.dot(x_tile, w_tile, out_dtype=tl.float32)
    acc += bias[None, :]
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    valid = mask_p[:, None] & mask_oc[None, :]
    gelu = tl.where(valid, gelu, 0.0)
    partial = tl.sum(gelu, axis=0)  # (BLOCK_OC,)

    out_off = (pid_n * NPIX_TILES + pid_pt) * BLOCK_OC + offs_oc
    tl.store(partial_ptr + out_off, partial, mask=mask_oc)


@triton.jit
def reduce_partial_kernel(
    partial_ptr, out_ptr,
    N, OC, NPIX_TILES,
    inv_npix,
    BLOCK_OC: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    for t_start in range(0, NPIX_TILES, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < NPIX_TILES
        # partial[(pid_n * NPIX_TILES + t) * BLOCK_OC + oc]
        ptrs = partial_ptr + (pid_n * NPIX_TILES + offs_t[:, None]) * BLOCK_OC + offs_oc[None, :]
        vals = tl.load(ptrs, mask=mask_t[:, None] & mask_oc[None, :], other=0.0)
        acc += tl.sum(vals, axis=0)

    result = acc * inv_npix
    tl.store(out_ptr + pid_n * OC + offs_oc, result, mask=mask_oc)


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

        K_CONST = IC * KH * KW
        K_PAD = 1
        while K_PAD < max(K_CONST, 16):
            K_PAD *= 2

        # BLOCK_OC = next pow2 >= OC, at least 16
        BLOCK_OC = 1
        while BLOCK_OC < max(OC, 16):
            BLOCK_OC *= 2

        out = torch.empty((N, OC), device=x.device, dtype=torch.float32)

        # We need NPIX_TILES which depends on BLOCK_PIX (autotuned).
        # Use a fixed sentinel: launch with grid=lambda meta and allocate partial with max tiles.
        # Allocate partial buffer sized to smallest BLOCK_PIX so it covers all configs.
        MIN_BLOCK_PIX = 64
        max_tiles = (npix + MIN_BLOCK_PIX - 1) // MIN_BLOCK_PIX
        partial = torch.empty((N, max_tiles, BLOCK_OC), device=x.device, dtype=torch.float32)

        def grid(meta):
            return (N, triton.cdiv(npix, meta['BLOCK_PIX']))

        # We pass NPIX_TILES based on the actual chosen BLOCK_PIX after the fact;
        # but since the kernel uses NPIX_TILES only as stride for output, we need
        # to know it at launch. Use a closure approach via lambda for kernel call.
        # Simpler: compute kernel-side via a small wrapper that picks BLOCK_PIX,
        # but autotune handles that. We make NPIX_TILES a kernel arg passed via lambda using meta.

        # Approach: do not autotune across BLOCK_PIX values that change NPIX_TILES;
        # instead pick a single BLOCK_PIX manually.
        BLOCK_PIX = 128
        NPIX_TILES = (npix + BLOCK_PIX - 1) // BLOCK_PIX

        partial = torch.empty((N, NPIX_TILES, BLOCK_OC), device=x.device, dtype=torch.float32)

        conv_gelu_avgpool_partial_kernel[(N, NPIX_TILES)](
            x, w, b, partial,
            N, IC, H, W,
            OC, OH, OW,
            NPIX_TILES,
            K_CONST=K_CONST,
            K_PAD=K_PAD,
            KH_CONST=KH,
            KW_CONST=KW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_PIX=BLOCK_PIX,
        )

        # Reduce
        BLOCK_T = 64
        while BLOCK_T < NPIX_TILES and BLOCK_T < 512:
            BLOCK_T *= 2
        if BLOCK_T > NPIX_TILES:
            # use a power of two >= NPIX_TILES up to cap
            BLOCK_T = 1
            while BLOCK_T < NPIX_TILES:
                BLOCK_T *= 2

        reduce_partial_kernel[(N,)](
            partial, out,
            N, OC, NPIX_TILES,
            inv_npix,
            BLOCK_OC=BLOCK_OC,
            BLOCK_T=min(BLOCK_T, 1024),
            num_warps=4,
        )

        return out