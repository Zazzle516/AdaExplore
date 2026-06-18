import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC_KH_KW'],
)
@triton.jit
def conv_scale_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    SCALE,
    OUT_HW, IC_KH_KW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile along output spatial (OH*OW)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial output positions
    m_mask = offs_m < OUT_HW

    oh = offs_m // OW
    ow = offs_m % OW

    # We will loop over OC in tiles of BLOCK_N, computing the conv output for each
    # (batch=pid_n, oc tile, spatial tile), scale and track min across oc.

    INF = float('inf')
    min_acc = tl.full([BLOCK_M], INF, dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    num_oc_tiles = (OC + BLOCK_N - 1) // BLOCK_N
    for oc_tile in range(0, num_oc_tiles):
        offs_n = oc_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < OC

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Loop over reduction dim IC*KH*KW
        for k_start in range(0, IC_KH_KW, BLOCK_K):
            k_idx = k_start + offs_k  # [BLOCK_K]
            k_mask = k_idx < IC_KH_KW

            ic = k_idx // (KH * KW)
            khw = k_idx % (KH * KW)
            kh = khw // KW
            kw = khw % KW

            # x indices: [BLOCK_M, BLOCK_K]
            # x[pid_n, ic, oh+kh, ow+kw]
            ih = oh[:, None] + kh[None, :]
            iw = ow[:, None] + kw[None, :]
            ic_b = ic[None, :]

            x_offset = (
                pid_n * IC * H * W
                + ic_b * (H * W)
                + ih * W
                + iw
            )
            x_load_mask = m_mask[:, None] & k_mask[None, :]
            x_vals = tl.load(x_ptr + x_offset, mask=x_load_mask, other=0.0)

            # w[oc, ic, kh, kw]: [BLOCK_K, BLOCK_N]
            w_offset = (
                offs_n[None, :] * (IC * KH * KW)
                + ic[:, None] * (KH * KW)
                + kh[:, None] * KW
                + kw[:, None]
            )
            w_load_mask = k_mask[:, None] & n_mask[None, :]
            w_vals = tl.load(w_ptr + w_offset, mask=w_load_mask, other=0.0)

            acc += tl.dot(x_vals, w_vals)

        # add bias
        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc + bias[None, :]
        acc = acc * SCALE

        # mask out invalid oc lanes to +inf so they don't affect min
        acc = tl.where(n_mask[None, :], acc, INF)

        tile_min = tl.min(acc, axis=1)  # [BLOCK_M]
        min_acc = tl.minimum(min_acc, tile_min)

    # Store output [N, 1, OH, OW]
    out_offset = pid_n * OUT_HW + offs_m
    tl.store(out_ptr + out_offset, min_acc, mask=m_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = float(scale_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, H, W = x.shape
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        OUT_HW = OH * OW
        IC_KH_KW = IC * KH * KW

        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (N, (OUT_HW + meta['BLOCK_M'] - 1) // meta['BLOCK_M'])

        conv_scale_min_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            self.scale_factor,
            OUT_HW, IC_KH_KW,
        )
        return out