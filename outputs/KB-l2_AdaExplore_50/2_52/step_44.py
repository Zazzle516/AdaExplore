import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _conv_configs():
    cfgs = []
    for (bm, bn, bk) in [
        (64, 128, 64), (128, 64, 64), (64, 64, 64),
        (128, 128, 32), (64, 128, 32), (128, 64, 32),
        (32, 128, 64), (128, 32, 64),
    ]:
        for nw in [4, 8]:
            ns = 2
            smem = (bm * bk + bk * bn) * 4 * ns
            if smem > 96 * 1024:
                continue
            cfgs.append(triton.Config(
                {'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk},
                num_warps=nw, num_stages=ns))
    return cfgs


@triton.autotune(configs=_conv_configs(), key=['N_HW', 'OC', 'IC', 'KH', 'KW'])
@triton.jit
def _conv_mish_bn_kernel(
    x_ptr,         # NHWC, contiguous, shape (N, H, W, IC)
    w_ptr,         # (OC, KH*KW*IC) with layout kh,kw,ic
    bias_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,       # NCHW, shape (N, OC, OH, OW)
    N, H, W, IC: tl.constexpr,
    OH, OW, OC,
    KH: tl.constexpr, KW: tl.constexpr,
    N_HW,
    K_TOTAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ow_idx = offs_m % OW
    tmp = offs_m // OW
    oh_idx = tmp % OH
    n_idx = tmp // OH

    m_mask = offs_m < N_HW
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    HWIC = H * W * IC
    WIC = W * IC
    n_base = n_idx * HWIC  # (BM,)

    # Fused K loop over KH*KW*IC
    for k_start in tl.static_range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # decompose k -> (kh, kw, ic)
        ic_k = offs_k % IC
        tmp_k = offs_k // IC
        kw_k = tmp_k % KW
        kh_k = tmp_k // KW

        ih_k = oh_idx[:, None] + kh_k[None, :]
        iw_k = ow_idx[:, None] + kw_k[None, :]

        x_idx = n_base[:, None] + ih_k * WIC + iw_k * IC + ic_k[None, :]
        x_tile = tl.load(x_ptr + x_idx, mask=m_mask[:, None], other=0.0)

        w_idx = offs_n[:, None] * K_TOTAL + offs_k[None, :]
        w_tile = tl.load(w_ptr + w_idx, mask=n_mask[:, None], other=0.0)

        acc += tl.dot(x_tile, tl.trans(w_tile))

    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
    shift = tl.load(shift_ptr + offs_n, mask=n_mask, other=0.0)

    x = acc + bias[None, :]
    sp = tl.log(1.0 + tl.exp(-tl.abs(x))) + tl.maximum(x, 0.0)
    th = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    y = x * th
    out = y * scale[None, :] + shift[None, :]

    # NCHW store: out[n, oc, oh, ow]
    OHW = OH * OW
    spatial = oh_idx * OW + ow_idx  # (BM,)
    out_idx = (n_idx[:, None] * (OC * OHW)
               + offs_n[None, :] * OHW
               + spatial[:, None])
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_idx, out, mask=store_mask)


def conv_mish_bn(x, weight, bias, scale, shift):
    N, IC, H, W = x.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    w_r = weight.permute(0, 2, 3, 1).contiguous().view(OC, KH * KW * IC)

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    N_HW = N * OH * OW
    K_TOTAL = KH * KW * IC

    grid = lambda META: (
        triton.cdiv(N_HW, META['BLOCK_M']),
        triton.cdiv(OC, META['BLOCK_N']),
    )

    _conv_mish_bn_kernel[grid](
        x_nhwc, w_r, bias, scale, shift, out,
        N, H, W, IC, OH, OW, OC,
        KH, KW, N_HW, K_TOTAL,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.training:
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
        bias = self.conv.bias
        if bias is None:
            bias = torch.zeros(self.conv.out_channels, device=x.device, dtype=x.dtype)
        return conv_mish_bn(x.contiguous(), self.conv.weight, bias.contiguous(),
                            scale, shift)