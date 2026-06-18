import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['N', 'OC', 'IC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv2d_mish_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    # x strides (NHWC: N, IH, IW, IC)
    x_sn, x_sh, x_sw, x_sc,
    # w strides (OC, KH, KW, IC)
    w_so, w_sh, w_sw, w_sc,
    # out strides (NHWC: N, OH, OW, OC)
    o_sn, o_sh, o_sw, o_sc,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_m = tl.program_id(1)  # output spatial tile (M = OH*OW)
    pid_oc = tl.program_id(2)  # OC tile

    M = OH * OW
    K = IC * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output spatial idx
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # output channel idx
    offs_k = tl.arange(0, BLOCK_K)

    oh = offs_m // OW
    ow = offs_m % OW
    m_mask = offs_m < M
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # base x pointer for this batch
    x_base = x_ptr + pid_n * x_sn

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K

        # decompose k into (ic, kh, kw)
        ic = k_idx // (KH * KW)
        khw = k_idx % (KH * KW)
        kh = khw // KW
        kw = khw % KW

        # x[N, ih, iw, ic] where ih=oh+kh, iw=ow+kw (no padding)
        ih = oh[:, None] + kh[None, :]  # [BLOCK_M, BLOCK_K]
        iw = ow[:, None] + kw[None, :]
        x_offs = ih * x_sh + iw * x_sw + ic[None, :] * x_sc
        x_mask_full = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_base + x_offs, mask=x_mask_full, other=0.0)

        # w[OC, kh, kw, ic] -> [BLOCK_K, BLOCK_N]
        w_offs = offs_n[None, :] * w_so + kh[:, None] * w_sh + kw[:, None] * w_sw + ic[:, None] * w_sc
        w_mask_full = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=w_mask_full, other=0.0)

        acc += tl.dot(x_tile, w_tile)

    # bias
    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b[None, :]

    # double mish: y = x * tanh(softplus(x)); applied twice
    # softplus(x) = log(1 + exp(x)); tanh = 2*sigmoid(2*sp) - 1
    sp1 = tl.log(1.0 + tl.exp(acc))
    t1 = 2.0 * tl.sigmoid(2.0 * sp1) - 1.0
    y1 = acc * t1
    sp2 = tl.log(1.0 + tl.exp(y1))
    t2 = 2.0 * tl.sigmoid(2.0 * sp2) - 1.0
    y2 = y1 * t2

    # store NHWC
    out_base = out_ptr + pid_n * o_sn
    out_offs = oh[:, None] * o_sh + ow[:, None] * o_sw + offs_n[None, :] * o_sc
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_base + out_offs, y2, mask=out_mask)


def conv2d_mish_mish(x, weight, bias):
    # x: NCHW float32
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    # convert to channels-last
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    w_ohwi = weight.permute(0, 2, 3, 1).contiguous()
    out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        N,
        triton.cdiv(OH * OW, META['BLOCK_M']),
        triton.cdiv(OC, META['BLOCK_N']),
    )

    conv2d_mish_mish_kernel[grid](
        x_nhwc, w_ohwi, bias, out_nhwc,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        w_ohwi.stride(0), w_ohwi.stride(1), w_ohwi.stride(2), w_ohwi.stride(3),
        out_nhwc.stride(0), out_nhwc.stride(1), out_nhwc.stride(2), out_nhwc.stride(3),
    )

    return out_nhwc.permute(0, 3, 1, 2).contiguous()


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return conv2d_mish_mish(x.contiguous(), self.conv.weight, self.conv.bias)