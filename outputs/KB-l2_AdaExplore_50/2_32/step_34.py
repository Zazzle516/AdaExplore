import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 2, 'BLOCK_OW': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 2, 'BLOCK_OW': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 16, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC_KHW'],
)
@triton.jit
def conv_scale_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    scale,
    IC_KHW,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    offs_oh = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)
    offs_ow = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_oh = offs_oh < OH
    mask_ow = offs_ow < OW
    mask_n = offs_n < OC

    BLOCK_M: tl.constexpr = BLOCK_OH * BLOCK_OW

    # flatten output spatial tile
    oh_flat = tl.reshape(offs_oh[:, None] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32), (BLOCK_M,))
    ow_flat = tl.reshape(tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32) + offs_ow[None, :], (BLOCK_M,))
    mask_m = tl.reshape(mask_oh[:, None] & mask_ow[None, :], (BLOCK_M,))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHW = KH * KW
    x_base = pid_n * IC * H * W

    for k0 in range(0, IC_KHW, BLOCK_K):
        k_idx = k0 + offs_k
        k_mask = k_idx < IC_KHW

        ic = k_idx // KHW
        rem = k_idx % KHW
        kh = rem // KW
        kw = rem % KW

        h_in = oh_flat[:, None] + kh[None, :]
        w_in = ow_flat[:, None] + kw[None, :]

        x_offset = x_base + ic[None, :] * (H * W) + h_in * W + w_in
        x_mask = mask_m[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        w_offset = offs_n[None, :] * IC_KHW + k_idx[:, None]
        w_mask = k_mask[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    acc = acc * scale

    INF = float('inf')
    acc = tl.where(mask_n[None, :], acc, INF)

    min_val = tl.min(acc, axis=1)  # [BLOCK_M]

    out_offset = pid_n * (OH * OW) + oh_flat * OW + ow_flat
    tl.store(out_ptr + out_offset, min_val, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
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
        IC_KHW = IC * KH * KW

        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_N = 1
        while BLOCK_N < OC:
            BLOCK_N *= 2

        grid = lambda meta: (
            N,
            triton.cdiv(OH, meta['BLOCK_OH']),
            triton.cdiv(OW, meta['BLOCK_OW']),
        )

        conv_scale_min_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            float(self.scale_factor),
            IC_KHW,
            BLOCK_N=BLOCK_N,
        )

        return out