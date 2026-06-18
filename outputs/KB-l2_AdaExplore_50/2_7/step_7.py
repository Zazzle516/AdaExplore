import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


def _conv_configs():
    configs = []
    for bm, bk in [(32, 16), (64, 16), (128, 16), (64, 32), (128, 32), (64, 54), (128, 54), (32, 32), (32, 54)]:
        for w in [2, 4, 8]:
            for s in [2, 3, 4]:
                configs.append(triton.Config({'BLOCK_M': bm, 'BLOCK_N': 32, 'BLOCK_K': bk}, num_warps=w, num_stages=s))
    return configs


@triton.autotune(configs=_conv_configs(), key=['OC', 'M', 'K'])
@triton.jit
def conv3d_implicit_gemm_kernel(
    x_ptr,        # [N, IC, ID, IH, IW]
    w_ptr,        # [OC, IC*KD*KH*KW]
    cbias_ptr,    # [OC] (conv bias)
    bias_ptr,     # [OC] (extra bias)
    out_ptr,      # [N, OC, OD, OH, OW]
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    M, K,  # M = OD*OH*OW, K = IC*KD*KH*KW
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)        # batch
    pid_m = tl.program_id(1)        # spatial tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # spatial pos
    offs_n = tl.arange(0, BLOCK_N)                     # OC indices (BLOCK_N >= OC)

    m_mask = offs_m < M
    n_mask = offs_n < OC

    # decompose output spatial
    od = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHW = KH * KW
    KDHW = KD * KHW

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # decompose K -> ic, kd, kh, kw
        ic = offs_k // KDHW
        rem_k = offs_k % KDHW
        kd = rem_k // KHW
        rem_k2 = rem_k % KHW
        kh = rem_k2 // KW
        kw = rem_k2 % KW

        # Compute input spatial indices for each (m, k)
        id_idx = od[:, None] + kd[None, :]   # [BM, BK]
        ih_idx = oh[:, None] + kh[None, :]
        iw_idx = ow[:, None] + kw[None, :]

        x_offs = (pid_n * stride_xn
                  + ic[None, :] * stride_xc
                  + id_idx * stride_xd
                  + ih_idx * stride_xh
                  + iw_idx * stride_xw)

        x_load_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_offs, mask=x_load_mask, other=0.0)

        # Load weights [BK, BN]: w[oc, k]
        w_offs = offs_n[None, :] * K + offs_k[:, None]
        w_mask = n_mask[None, :] & k_mask[:, None]
        w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # Add conv bias
    cb = tl.load(cbias_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + cb[None, :]

    # Fused activations: ReLU -> LeakyReLU(0.01) -> GELU -> Sigmoid -> +bias
    # After ReLU, values >= 0, so LeakyReLU is no-op.
    y = tl.maximum(acc, 0.0)
    inv_sqrt2 = 0.70710678118654752440
    y = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    y = tl.sigmoid(y)

    extra_b = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    y = y + extra_b[None, :]

    # Store output
    out_offs = (pid_n * stride_on
                + offs_n[None, :] * stride_oc
                + od[:, None] * stride_od
                + oh[:, None] * stride_oh
                + ow[:, None] * stride_ow)
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offs, y, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.KD = self.KH = self.KW = kernel_size
        else:
            self.KD, self.KH, self.KW = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD, KH, KW = self.KD, self.KH, self.KW
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        # Pack weight: conv weight is [OC, IC, KD, KH, KW]
        w = self.conv.weight.contiguous().view(OC, IC * KD * KH * KW)
        cb = self.conv.bias.contiguous()
        bias_flat = self.bias.contiguous().view(-1)

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        M = OD * OH * OW
        K = IC * KD * KH * KW

        sx = x.stride()
        so = out.stride()

        grid = lambda meta: (
            N,
            triton.cdiv(M, meta['BLOCK_M']),
        )

        conv3d_implicit_gemm_kernel[grid](
            x, w, cb, bias_flat, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            M, K,
            sx[0], sx[1], sx[2], sx[3], sx[4],
            so[0], so[1], so[2], so[3], so[4],
        )
        return out