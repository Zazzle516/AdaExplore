import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_nhwc_fused_kernel(
    x_ptr,      # NHWC input  [N, H, W, IC]
    w_ptr,      # weights [OC, KH, KW, IC]  (flattened K = KH*KW*IC, K-major last)
    bias_ptr,   # [OC]  fused bias = conv_bias * multiplier
    scale_ptr,  # [OC]  = multiplier
    out_ptr,    # NHWC output [N, OH, OW, OC]
    N, H, W, IC,
    OH, OW, OC,
    KH: tl.constexpr, KW: tl.constexpr,
    M, NN, K,    # GEMM dims: M=N*OH*OW, NN=OC, K=KH*KW*IC
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output spatial linear index (n*OH*OW + oh*OW + ow)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC

    # decode m -> (n, oh, ow)
    ohw = OH * OW
    n_idx = offs_m // ohw
    rem = offs_m % ohw
    oh_idx = rem // OW
    ow_idx = rem % OW

    m_mask = offs_m < M
    n_mask = offs_n < NN

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over KH * KW * IC, K-tiles of size BLOCK_K
    # weights laid out as [OC, KH*KW*IC] with last dim contiguous
    offs_k = tl.arange(0, BLOCK_K)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k          # [BLOCK_K]
        k_valid = k_idx < K

        # decode k -> (kh, kw, ic)
        kh_idx = k_idx // (KW * IC)
        kwic = k_idx % (KW * IC)
        kw_idx = kwic // IC
        ic_idx = kwic % IC

        # input H/W index per (m, k)
        ih = oh_idx[:, None] + kh_idx[None, :]  # [BLOCK_M, BLOCK_K]
        iw = ow_idx[:, None] + kw_idx[None, :]

        # input ptr: x[n, ih, iw, ic]
        x_offs = (n_idx[:, None] * (H * W * IC)
                  + ih * (W * IC)
                  + iw * IC
                  + ic_idx[None, :])
        x_m = m_mask[:, None] & k_valid[None, :]
        x_tile = tl.load(x_ptr + x_offs, mask=x_m, other=0.0)  # [BLOCK_M, BLOCK_K]

        # weight ptr: w[oc, k]   layout [OC, K]
        w_offs = offs_n[:, None] * K + k_idx[None, :]   # [BLOCK_N, BLOCK_K]
        w_m = n_mask[:, None] & k_valid[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=w_m, other=0.0)  # [BLOCK_N, BLOCK_K]

        acc += tl.dot(x_tile, tl.trans(w_tile))

    # epilogue: bias_fused (= conv_bias * multiplier), scale (= multiplier)
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)  # [BLOCK_N]
    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)    # [BLOCK_N]

    y = acc * scale[None, :] + bias[None, :]
    # LeakyReLU(0.01)
    y = tl.where(y >= 0, y, y * 0.01)
    # exact GELU
    inv_sqrt2 = 0.7071067811865475
    y = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    # store to NHWC output [N, OH, OW, OC]
    out_offs = (n_idx[:, None] * (OH * OW * OC)
                + oh_idx[:, None] * (OW * OC)
                + ow_idx[:, None] * OC
                + offs_n[None, :])
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offs, y, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        N, C, H, W_ = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = H - KH + 1
        OW = W_ - KW + 1

        # Convert input to NHWC contiguous
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        # Weight: conv weight is [OC, IC, KH, KW] -> [OC, KH, KW, IC]
        w = self.conv.weight  # [OC, IC, KH, KW]
        w_nhwc = w.permute(0, 2, 3, 1).contiguous().view(OC, KH * KW * C)

        mult_flat = self.multiplier.contiguous().view(-1)  # [OC]
        conv_bias = self.conv.bias  # [OC]
        fused_bias = conv_bias * mult_flat  # [OC]

        out = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        M = N * OH * OW
        NN = OC
        K = KH * KW * C

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(NN, meta['BLOCK_N']),
        )

        conv2d_nhwc_fused_kernel[grid](
            x_nhwc, w_nhwc, fused_bias, mult_flat, out,
            N, H, W_, C,
            OH, OW, OC,
            KH, KW,
            M, NN, K,
        )

        # NHWC -> NCHW
        out = out.permute(0, 3, 1, 2).contiguous()
        return out