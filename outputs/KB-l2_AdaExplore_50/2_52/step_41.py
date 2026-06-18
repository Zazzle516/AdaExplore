import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Conv2d (3x3, stride 1, no pad) implemented as implicit im2col GEMM with
# Mish activation + folded BN scale/shift in epilogue.
# Layout: input is NHWC (channels_last), weight is (OC, KH*KW*IC) row-major.

def _conv_configs():
    cfgs = []
    for bm in [32, 64, 128]:
        for bn in [32, 64, 128]:
            for bk in [16, 32]:
                for nw in [4, 8]:
                    # SMEM check
                    smem = (bm * bk + bk * bn) * 4 * 2
                    if smem > 96 * 1024:
                        continue
                    if bm * bn > 128 * 128:
                        continue
                    cfgs.append(triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk},
                                              num_warps=nw, num_stages=2))
    return cfgs


@triton.autotune(configs=_conv_configs(), key=['N_HW', 'OC', 'K_TOTAL'])
@triton.jit
def _conv3x3_mish_bn_kernel(
    x_ptr,         # NHWC, contiguous, shape (N, H, W, IC)
    w_ptr,         # (OC, KH*KW*IC)
    bias_ptr,      # (OC,)
    scale_ptr,     # (OC,)
    shift_ptr,     # (OC,)
    out_ptr,       # NHWC, shape (N, OH, OW, OC)
    N, H, W, IC,
    OH, OW, OC,
    KH: tl.constexpr, KW: tl.constexpr,
    N_HW,          # N * OH * OW
    K_TOTAL,       # KH * KW * IC
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # along N*OH*OW
    pid_n = tl.program_id(1)  # along OC

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC indices

    # decode m -> (n, oh, ow)
    ow_idx = offs_m % OW
    tmp = offs_m // OW
    oh_idx = tmp % OH
    n_idx = tmp // OH

    m_mask = offs_m < N_HW
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K dim = KH * KW * IC; we iterate in BLOCK_K chunks.
    # For each k, decode k -> (kh, kw, ic), compute input index.
    # input stride (NHWC): n * H*W*IC + h * W*IC + w * IC + ic
    HWIC = H * W * IC
    WIC = W * IC

    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_TOTAL

        ic_k = offs_k % IC
        tmp2 = offs_k // IC
        kw_k = tmp2 % KW
        kh_k = tmp2 // KW

        # input positions
        ih = oh_idx[:, None] + kh_k[None, :]  # (BM, BK)
        iw = ow_idx[:, None] + kw_k[None, :]
        ic_b = ic_k[None, :]  # (1, BK)
        n_b = n_idx[:, None]  # (BM, 1)

        x_idx = n_b * HWIC + ih * WIC + iw * IC + ic_b  # (BM, BK)
        x_load_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_idx, mask=x_load_mask, other=0.0)  # (BM, BK)

        # weight: (OC, K_TOTAL) -> w[oc, k]
        w_idx = offs_n[:, None] * K_TOTAL + offs_k[None, :]  # (BN, BK)
        w_load_mask = n_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_idx, mask=w_load_mask, other=0.0)  # (BN, BK)

        acc += tl.dot(x_tile, tl.trans(w_tile))

    # epilogue: bias, mish, bn scale/shift
    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
    shift = tl.load(shift_ptr + offs_n, mask=n_mask, other=0.0)

    x = acc + bias[None, :]
    # softplus: log(1+exp(x)); stable form
    sp = tl.log(1.0 + tl.exp(-tl.abs(x))) + tl.maximum(x, 0.0)
    th = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    y = x * th
    out = y * scale[None, :] + shift[None, :]

    # store NHWC: out[n, oh, ow, oc]
    OHWOC = OH * OW * OC
    OWOC = OW * OC
    out_idx = n_idx[:, None] * OHWOC + oh_idx[:, None] * OWOC + ow_idx[:, None] * OC + offs_n[None, :]
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_idx, out, mask=store_mask)


def conv3x3_mish_bn(x, weight, bias, scale, shift):
    # x: (N, IC, H, W) NCHW
    # weight: (OC, IC, KH, KW)
    N, IC, H, W = x.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    # to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # weight to (OC, KH*KW*IC) layout matching k-decode (k -> kh,kw,ic)
    w_r = weight.permute(0, 2, 3, 1).contiguous().view(OC, KH * KW * IC)

    out = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

    N_HW = N * OH * OW
    K_TOTAL = KH * KW * IC

    grid = lambda META: (
        triton.cdiv(N_HW, META['BLOCK_M']),
        triton.cdiv(OC, META['BLOCK_N']),
    )

    _conv3x3_mish_bn_kernel[grid](
        x_nhwc, w_r, bias, scale, shift, out,
        N, H, W, IC, OH, OW, OC,
        KH, KW, N_HW, K_TOTAL,
    )

    # back to NCHW
    return out.permute(0, 3, 1, 2).contiguous()


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.training or self.kernel_size != 3:
            y = self.conv(x)
            y = torch.multiply(torch.tanh(F.softplus(y)), y)
            y = self.bn(y)
            return y
        rm = self.bn.running_mean
        rv = self.bn.running_var
        w = self.bn.weight
        b = self.bn.bias
        invstd = torch.rsqrt(rv + self.eps)
        scale = (w * invstd).contiguous()
        shift = (b - rm * w * invstd).contiguous()
        return conv3x3_mish_bn(x.contiguous(),
                               self.conv.weight,
                               self.conv.bias,
                               scale, shift)