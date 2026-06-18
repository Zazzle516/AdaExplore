import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = OH * OW
    K = IC * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    oh = offs_m // OW
    ow = offs_m % OW

    running_min = tl.full((BLOCK_M,), float('inf'), dtype=tl.float32)

    num_oc_tiles = tl.cdiv(OC, BLOCK_N)

    for oc_tile in range(0, num_oc_tiles):
        offs_n = oc_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < OC

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            ic = offs_k // (KH * KW)
            khw = offs_k % (KH * KW)
            kh = khw // KW
            kw = khw % KW

            # NHWC input: x[n, ih, iw, ic]
            ih = oh[:, None] + kh[None, :]
            iw = ow[:, None] + kw[None, :]
            ic_b = tl.broadcast_to(ic[None, :], (BLOCK_M, BLOCK_K))

            x_offset = (
                pid_n * IH * IW * IC
                + ih * (IW * IC)
                + iw * IC
                + ic_b
            )
            x_valid = m_mask[:, None] & k_mask[None, :]
            x_vals = tl.load(x_ptr + x_offset, mask=x_valid, other=0.0)

            # Weight layout: (OC, KH, KW, IC) -> index [oc, kh, kw, ic]
            # offs_k decomposes as ic*KH*KW + kh*KW + kw
            # In new layout: kh*KW*IC + kw*IC + ic
            w_k_offset = kh * (KW * IC) + kw * IC + ic  # [BLOCK_K]
            w_offset = offs_n[:, None] * (KH * KW * IC) + w_k_offset[None, :]
            w_valid = n_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptr + w_offset, mask=w_valid, other=0.0)

            acc += tl.dot(x_vals, tl.trans(w_vals))

        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc + bias[None, :]
        acc = acc * scale

        acc = tl.where(n_mask[None, :], acc, float('inf'))

        tile_min = tl.min(acc, axis=1)
        running_min = tl.minimum(running_min, tile_min)

    out_offset = pid_n * OH * OW + offs_m
    tl.store(out_ptr + out_offset, running_min, mask=m_mask)


def conv2d_scale_min(x_nhwc, weight_nhwc, bias, scale, N, IC, IH, IW, OC, KH, KW):
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, 1, OH, OW), device=x_nhwc.device, dtype=torch.float32)

    M = OH * OW
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), N)

    conv2d_scale_min_kernel[grid](
        x_nhwc, weight_nhwc, bias, out,
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to (OC, KH, KW, IC)
        w = self.conv.weight.detach().cuda().contiguous()
        w_nhwc = w.permute(0, 2, 3, 1).contiguous()
        self.register_buffer('weight_nhwc', w_nhwc)
        self.register_buffer('bias_cuda', self.conv.bias.detach().cuda().contiguous())

    def forward(self, x):
        x = x.cuda().contiguous()
        N, IC, IH, IW = x.shape
        # Permute to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        OC = self.out_channels
        KH = KW = self.kernel_size
        return conv2d_scale_min(
            x_nhwc, self.weight_nhwc, self.bias_cuda, self.scale_factor,
            N, IC, IH, IW, OC, KH, KW
        )