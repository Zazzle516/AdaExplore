import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_implicit_gemm_kernel(
    x_ptr, w_ptr, b_ptr, mult_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # M = OC, N = N*OH*OW, K = IC*KH*KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial+batch flat indices

    OHW = OH * OW
    # Decompose offs_n into (n, oh, ow)
    n_idx = offs_n // OHW
    rem = offs_n % OHW
    oh_idx = rem // OW
    ow_idx = rem % OW

    m_mask = offs_m < OC
    n_mask = offs_n < (N * OHW)

    K_total = IC * KH * KW
    KHW = KH * KW

    IH_IW = IH * IW
    IC_IH_IW = IC * IH_IW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_total

        # decompose k -> (ic, kh, kw)
        ic_k = offs_k // KHW
        kk = offs_k % KHW
        kh_k = kk // KW
        kw_k = kk % KW

        # Load W: shape (BLOCK_M, BLOCK_K). W layout [OC, IC, KH, KW]
        w_idx = offs_m[:, None] * K_total + offs_k[None, :]
        w_load_mask = m_mask[:, None] & k_mask[None, :]
        w = tl.load(w_ptr + w_idx, mask=w_load_mask, other=0.0)

        # Load X: shape (BLOCK_K, BLOCK_N).
        # x[n, ic, oh+kh, ow+kw]
        ih_idx = oh_idx[None, :] + kh_k[:, None]
        iw_idx = ow_idx[None, :] + kw_k[:, None]
        x_idx = (n_idx[None, :] * IC_IH_IW
                 + ic_k[:, None] * IH_IW
                 + ih_idx * IW
                 + iw_idx)
        x_load_mask = k_mask[:, None] & n_mask[None, :]
        x = tl.load(x_ptr + x_idx, mask=x_load_mask, other=0.0)

        acc = tl.dot(w, x, acc)

    # Bias + multiplier (per output channel)
    bias = tl.load(b_ptr + offs_m, mask=m_mask, other=0.0)
    mult = tl.load(mult_ptr + offs_m, mask=m_mask, other=0.0)
    acc = (acc + bias[:, None]) * mult[:, None]

    # LeakyReLU
    acc = tl.where(acc >= 0, acc, acc * 0.01)

    # GELU exact
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Store: out shape [N, OC, OH, OW] -> flat index n*OC*OHW + oc*OHW + (oh*OW+ow)
    out_idx = n_idx[None, :] * OC * OHW + offs_m[:, None] * OHW + rem[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


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

        M = OC
        Nn = N * OH * OW

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(Nn, META['BLOCK_N']))

        conv_implicit_gemm_kernel[grid](
            x, w, b, mult, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH=KH, KW=KW,
        )
        return out