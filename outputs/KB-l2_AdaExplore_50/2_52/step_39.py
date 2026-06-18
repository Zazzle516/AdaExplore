import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Conv2d (3x3, no padding, stride 1) implemented as implicit im2col GEMM
# with fused Mish + BN (folded) in the epilogue.
# Input layout: NHWC (channels_last), Weight: (OC, IC, KH, KW) -> reshape to (OC, IC*KH*KW)
# Output layout: NHWC

def _conv_configs():
    configs = []
    for bm in [32, 64, 128]:
        for bn in [32, 64, 128]:
            for bk in [16, 32]:
                for nw in [4, 8]:
                    for ns in [2, 3]:
                        # SMEM check: (bm*bk + bk*bn)*4*ns
                        smem = (bm * bk + bk * bn) * 4 * ns
                        if smem > 100 * 1024:
                            continue
                        if bm * bn < 1024:
                            continue
                        configs.append(triton.Config(
                            {'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk},
                            num_warps=nw, num_stages=ns))
    return configs


@triton.autotune(configs=_conv_configs(), key=['M', 'N', 'K'])
@triton.jit
def _conv2d_mish_bn_kernel(
    x_ptr,           # input NHWC: (N, H, W, IC)
    w_ptr,           # weight: (OC, IC*KH*KW)  -- reshaped (OC, IC, KH, KW)
    b_ptr,           # conv bias (OC,)
    scale_ptr,       # BN scale (OC,)
    shift_ptr,       # BN shift (OC,)
    out_ptr,         # output NHWC: (N, OH, OW, OC)
    N, IC, H, W, OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    M, NN, K,        # M = N*OH*OW, NN = OC, K = IC*KH*KW
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
    offs_k = tl.arange(0, BLOCK_K)                    # [BK]

    # decode m -> (n, oh, ow)
    ohw = OH * OW
    n_idx = offs_m // ohw
    rem_m = offs_m - n_idx * ohw
    oh_idx = rem_m // OW
    ow_idx = rem_m - oh_idx * OW

    m_mask = offs_m < M
    n_mask = offs_n < NN

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # x stride NHWC: x[n, h, w, ic] = x_ptr + n*H*W*IC + h*W*IC + w*IC + ic
    # iterate over K = IC*KH*KW
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k  # [BK]
        k_mask = k_offs < K
        # decode k -> (ic, kh, kw)
        ic_idx = k_offs // (KH * KW)
        rem_k = k_offs - ic_idx * (KH * KW)
        kh_idx = rem_k // KW
        kw_idx = rem_k - kh_idx * KW

        # input positions: ih = oh + kh, iw = ow + kw (no padding)
        ih = oh_idx[:, None] + kh_idx[None, :]  # [BM, BK]
        iw = ow_idx[:, None] + kw_idx[None, :]  # [BM, BK]

        x_offs = (n_idx[:, None] * (H * W * IC) +
                  ih * (W * IC) +
                  iw * IC +
                  ic_idx[None, :])  # [BM, BK]

        x_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

        # weight: w[oc, k] = w_ptr + oc*K + k  (OC, K)
        w_offs = offs_n[:, None] * K + k_offs[None, :]  # [BN, BK]
        w_mask = n_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BN, BK]

        acc += tl.dot(x_tile, tl.trans(w_tile))

    # add bias
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)  # [BN]
    acc = acc + bias[None, :]

    # Mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    th = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    y = acc * th

    # BN fused
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
    shift = tl.load(shift_ptr + offs_n, mask=n_mask, other=0.0)
    y = y * scale[None, :] + shift[None, :]

    # store NHWC: out[n, oh, ow, oc] at offset n*OH*OW*OC + oh*OW*OC + ow*OC + oc
    out_offs = (n_idx[:, None] * (OH * OW * OC) +
                oh_idx[:, None] * (OW * OC) +
                ow_idx[:, None] * OC +
                offs_n[None, :])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offs, y, mask=out_mask)


def conv2d_mish_bn(x_nhwc, w_flat, bias, scale, shift, N, H, W, IC, OC, OH, OW, KH, KW):
    M = N * OH * OW
    NN = OC
    K = IC * KH * KW
    out = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(NN, meta['BLOCK_N']))
    _conv2d_mish_bn_kernel[grid](
        x_nhwc, w_flat, bias, scale, shift, out,
        N, IC, H, W, OC, OH, OW,
        KH, KW,
        M, NN, K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.training:
            x = self.conv(x)
            x = torch.multiply(torch.tanh(F.softplus(x)), x)
            x = self.bn(x)
            return x
        else:
            N, IC, H, W = x.shape
            KH = KW = self.kernel_size
            OH = H - KH + 1
            OW = W - KW + 1
            OC = self.out_channels

            # NHWC input
            x_nhwc = x.permute(0, 2, 3, 1).contiguous()

            # weight: (OC, IC, KH, KW) -> need (OC, IC*KH*KW) where K decodes as (ic, kh, kw)
            w = self.conv.weight.contiguous().view(OC, IC * KH * KW).contiguous()
            bias = self.conv.bias.contiguous()

            rm = self.bn.running_mean
            rv = self.bn.running_var
            bw = self.bn.weight
            bb = self.bn.bias
            invstd = torch.rsqrt(rv + self.eps)
            scale = (bw * invstd).contiguous()
            shift = (bb - rm * bw * invstd).contiguous()

            out_nhwc = conv2d_mish_bn(x_nhwc, w, bias, scale, shift,
                                      N, H, W, IC, OC, OH, OW, KH, KW)
            out = out_nhwc.permute(0, 3, 1, 2).contiguous()
            return out