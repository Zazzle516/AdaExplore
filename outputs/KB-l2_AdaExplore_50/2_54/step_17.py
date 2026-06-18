import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'IC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv2d_fused_kernel(
    x_ptr,         # [N, H, W, IC] NHWC
    w_ptr,         # [OC, KH*KW*IC]
    b_ptr,         # [OC]
    m_ptr,         # [OC]
    out_ptr,       # [N, OH, OW, OC] NHWC
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids
    pid_n_oc = tl.program_id(0)  # OC tile
    pid_sp = tl.program_id(1)    # spatial tile (over OH*OW)
    pid_batch = tl.program_id(2) # batch

    oc_start = pid_n_oc * BLOCK_M
    sp_start = pid_sp * BLOCK_N

    offs_oc = oc_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_sp = sp_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    oh = offs_sp // OW
    ow = offs_sp % OW

    mask_oc = offs_oc < OC
    mask_sp = offs_sp < (OH * OW)

    K = IC * KH * KW

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # iterate over K dimension: KH * KW * IC
    # We loop over kh, kw, then IC in BLOCK_K chunks
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # padding=0
            iw = ow + kw
            # base x offset for this (kh, kw): x[batch, ih, iw, :ic]
            # x layout: [N, H, W, IC], stride: (H*W*IC, W*IC, IC, 1)
            x_base = pid_batch * H * W * IC + ih * W * IC + iw * IC  # [BLOCK_N]
            # w offset for this kh, kw: w[oc, kh*KW*IC + kw*IC + ic]
            w_base_kk = (kh * KW + kw) * IC  # scalar K offset

            for ic_start in range(0, IC, BLOCK_K):
                offs_k = ic_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
                mask_k = offs_k < IC

                # load x: [BLOCK_N, BLOCK_K]
                x_addrs = x_base[:, None] + offs_k[None, :]
                x_mask = mask_sp[:, None] & mask_k[None, :]
                x_vals = tl.load(x_ptr + x_addrs, mask=x_mask, other=0.0)

                # load w: [BLOCK_M, BLOCK_K]
                w_addrs = offs_oc[:, None] * K + (w_base_kk + offs_k[None, :])
                w_mask = mask_oc[:, None] & mask_k[None, :]
                w_vals = tl.load(w_ptr + w_addrs, mask=w_mask, other=0.0)

                # GEMM: acc[BLOCK_M, BLOCK_N] += w[BLOCK_M, BLOCK_K] @ x[BLOCK_N, BLOCK_K].T
                acc += tl.dot(w_vals, tl.trans(x_vals), allow_tf32=True)

    # bias
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)  # [BLOCK_M]
    mult = tl.load(m_ptr + offs_oc, mask=mask_oc, other=0.0)  # [BLOCK_M]

    acc = acc + bias[:, None]
    acc = acc * mult[:, None]

    # leaky relu
    acc = tl.where(acc >= 0, acc, acc * 0.01)
    # gelu
    inv_sqrt2 = 0.70710678118654752440
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # store: out[batch, oh, ow, oc] in NHWC
    # store as [BLOCK_N, BLOCK_M] (sp, oc) - need transpose
    out_acc = tl.trans(acc)  # [BLOCK_N, BLOCK_M]
    out_addrs = pid_batch * OH * OW * OC + offs_sp[:, None] * OC + offs_oc[None, :]
    out_mask = mask_sp[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_addrs, out_acc, mask=out_mask)


def conv2d_fused(x_nhwc, w_flat, bias, multiplier, N, IC, H, W, OC, OH, OW, KH, KW):
    out = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(OC, meta['BLOCK_M']),
        triton.cdiv(OH * OW, meta['BLOCK_N']),
        N,
    )
    conv2d_fused_kernel[grid](
        x_nhwc, w_flat, bias, multiplier, out,
        N, IC, H, W,
        OC, OH, OW,
        KH, KW,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, H, W = x.shape
        KH = self.kernel_size
        KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        OC = self.out_channels

        # NHWC input
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        # weight [OC, IC, KH, KW] -> [OC, KH, KW, IC] -> [OC, KH*KW*IC]
        w = self.conv.weight.permute(0, 2, 3, 1).contiguous().view(OC, KH * KW * IC)
        b = self.conv.bias
        m = self.multiplier.contiguous().view(-1)

        out_nhwc = conv2d_fused(x_nhwc, w, b, m, N, IC, H, W, OC, OH, OW, KH, KW)

        # back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out