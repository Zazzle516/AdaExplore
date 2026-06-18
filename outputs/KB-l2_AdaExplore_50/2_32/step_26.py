import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['N', 'OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_scale_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program: one batch n, tile of output spatial (M = OH*OW), reduces over OC by min.
    pid_m = tl.program_id(0)  # output spatial tile
    pid_n = tl.program_id(1)  # batch

    M = OH * OW
    K = IC * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    # Decompose m -> (oh, ow)
    oh = offs_m // OW
    ow = offs_m % OW

    # We'll iterate over output channels in tiles of BLOCK_N, computing GEMM,
    # applying scale, then min-reducing into running min vector of length BLOCK_M.
    NEG_INF = float('inf')
    running_min = tl.full((BLOCK_M,), NEG_INF, dtype=tl.float32)

    num_oc_tiles = tl.cdiv(OC, BLOCK_N)

    for oc_tile in range(0, num_oc_tiles):
        offs_n = oc_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < OC

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop over reduction K = IC*KH*KW in BLOCK_K chunks
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            # Decompose k -> (ic, kh, kw)
            ic = offs_k // KHW
            khw = offs_k % KHW
            kh = khw // KW
            kw = khw % KW

            # Input gather: x[n, ic, oh+kh, ow+kw]
            ih = oh[:, None] + kh[None, :]  # [BLOCK_M, BLOCK_K]
            iw = ow[:, None] + kw[None, :]  # [BLOCK_M, BLOCK_K]
            ic_b = tl.broadcast_to(ic[None, :], (BLOCK_M, BLOCK_K))

            x_offset = (
                x_batch_off
                + ic_b * IHIW
                + ih * IW
                + iw
            )
            x_valid = m_mask[:, None] & k_mask[None, :] & (ih < IH) & (iw < IW)
            x_vals = tl.load(x_ptr + x_offset, mask=x_valid, other=0.0)

            # Weight: w[oc, ic, kh, kw]  shape [BLOCK_N, BLOCK_K]
            w_offset = (
                offs_n[:, None] * (IC * KH * KW)
                + offs_k[None, :]
            )
            w_valid = n_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptr + w_offset, mask=w_valid, other=0.0)

            # acc += x_vals @ w_vals.T
            acc += tl.dot(x_vals, tl.trans(w_vals))

        # Add bias
        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc + bias[None, :]
        acc = acc * scale

        # Mask invalid OC lanes to +inf so they don't affect min
        acc = tl.where(n_mask[None, :], acc, float('inf'))

        # Min along N dimension
        tile_min = tl.min(acc, axis=1)
        running_min = tl.minimum(running_min, tile_min)

    # Store output: out[n, 0, oh, ow]
    out_offset = pid_n * OH * OW + offs_m
    tl.store(out_ptr + out_offset, running_min, mask=m_mask)


def conv2d_scale_min(x, weight, bias, scale):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    out = torch.empty((N, 1, OH, OW), device=x.device, dtype=torch.float32)

    M = OH * OW
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), N)

    conv2d_scale_min_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        float(scale),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.cuda().contiguous()
        b = self.conv.bias.cuda().contiguous()
        return conv2d_scale_min(x, w, b, self.scale_factor)