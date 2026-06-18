import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_implicit_gemm_kernel(
    x_ptr, w_ptr, b_ptr, mult_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
    BLOCK_K: tl.constexpr,  # K reduction tile
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_sp = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_sp = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial indices

    oh = offs_sp // OW
    ow = offs_sp % OW

    m_mask = offs_m < OC
    sp_mask = offs_sp < (OH * OW)

    KHW = KH * KW
    K_TOTAL = IC * KHW
    IH_IW = IH * IW

    # x base for this batch
    x_batch_base = pid_n * IC * IH_IW

    # w stride: w is [OC, IC, KH, KW] -> offset = oc * IC*KH*KW + k
    # Iterate K dimension in chunks of BLOCK_K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_idx = k_start + offs_k  # (BLOCK_K,)
        k_mask = k_idx < K_TOTAL

        # decompose k_idx into (ic, kh, kw)
        ic = k_idx // KHW
        rem = k_idx % KHW
        kh = rem // KW
        kw = rem % KW

        # Load weight tile [BLOCK_M, BLOCK_K]
        w_offs = offs_m[:, None] * K_TOTAL + k_idx[None, :]
        w_mask_2d = m_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=w_mask_2d, other=0.0)

        # Load input tile [BLOCK_K, BLOCK_N]
        # x[n, ic, oh+kh, ow+kw]
        ih_idx = oh[None, :] + kh[:, None]  # (BLOCK_K, BLOCK_N)
        iw_idx = ow[None, :] + kw[:, None]
        x_offs = x_batch_base + ic[:, None] * IH_IW + ih_idx * IW + iw_idx
        x_mask_2d = k_mask[:, None] & sp_mask[None, :]
        x_tile = tl.load(x_ptr + x_offs, mask=x_mask_2d, other=0.0)

        acc += tl.dot(w_tile, x_tile)

    # epilogue: bias + multiplier + leakyrelu + gelu
    bias = tl.load(b_ptr + offs_m, mask=m_mask, other=0.0)
    mult = tl.load(mult_ptr + offs_m, mask=m_mask, other=0.0)
    acc = acc + bias[:, None]
    acc = acc * mult[:, None]
    acc = tl.where(acc >= 0, acc, acc * 0.01)
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    OH_OW = OH * OW
    out_offs = pid_n * OC * OH_OW + offs_m[:, None] * OH_OW + offs_sp[None, :]
    out_mask = m_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        mult = self.multiplier.contiguous().view(-1)

        N, IC, IH, IW = x.shape
        OC = w.shape[0]
        KH = w.shape[2]
        KW = w.shape[3]
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda META: (
            N,
            triton.cdiv(OC, META['BLOCK_M']),
            triton.cdiv(OH * OW, META['BLOCK_N']),
        )

        conv_implicit_gemm_kernel[grid](
            x, w, b, mult, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH=KH, KW=KW,
        )
        return out