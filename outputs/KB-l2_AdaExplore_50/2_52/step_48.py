import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _conv_configs():
    cfgs = []
    for bm, bn, bk, nw, ns in [
        (64, 128, 32, 8, 3),
        (128, 64, 32, 8, 3),
        (64, 128, 64, 8, 2),
        (128, 64, 64, 8, 2),
        (64, 64, 64, 4, 3),
        (128, 128, 32, 8, 2),
        (64, 256, 32, 8, 2),
        (256, 64, 32, 8, 2),
        (32, 128, 64, 4, 2),
        (128, 32, 64, 4, 2),
    ]:
        smem = (bm * bk + bk * bn) * 4 * ns
        if smem > 100 * 1024:
            continue
        cfgs.append(triton.Config(
            {'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk},
            num_warps=nw, num_stages=ns))
    return cfgs


@triton.autotune(configs=_conv_configs(), key=['N_HW', 'OC', 'IC'])
@triton.jit
def _conv3x3_mish_bn_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    N, H, W, IC,
    OH, OW, OC,
    N_HW,
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
    KHKW_IC = 9 * IC  # KH=KW=3

    # Flattened K loop over KHKW*IC, processing BLOCK_K elements per iter.
    # For each k in tile: kh = k // (3*IC), kw = (k // IC) % 3, ic = k % IC.
    # Input addr = n*HWIC + (oh+kh)*WIC + (ow+kw)*IC + ic
    # Weight (K-major) addr = k * OC + oc
    base_n = n_idx * HWIC  # (BM,)
    base_oh = oh_idx * WIC  # (BM,)
    base_ow = ow_idx * IC  # (BM,)

    for k_start in range(0, KHKW_IC, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < KHKW_IC

        ic_k = offs_k % IC
        khkw = offs_k // IC
        kw_k = khkw % 3
        kh_k = khkw // 3

        # Input index for each (m, k)
        x_idx = (base_n[:, None]
                 + (oh_idx[:, None] + kh_k[None, :]) * WIC
                 + (ow_idx[:, None] + kw_k[None, :]) * IC
                 + ic_k[None, :])
        x_load_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_idx, mask=x_load_mask, other=0.0)

        # Weight K-major: index = k * OC + oc
        w_idx = offs_k[:, None] * OC + offs_n[None, :]
        w_load_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_ptr + w_idx, mask=w_load_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile)

    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
    shift = tl.load(shift_ptr + offs_n, mask=n_mask, other=0.0)

    x = acc + bias[None, :]
    sp = tl.log(1.0 + tl.exp(-tl.abs(x))) + tl.maximum(x, 0.0)
    th = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    y = x * th
    out = y * scale[None, :] + shift[None, :]

    OHWOC = OH * OW * OC
    OWOC = OW * OC
    out_idx = (n_idx[:, None] * OHWOC
               + oh_idx[:, None] * OWOC
               + ow_idx[:, None] * OC
               + offs_n[None, :])
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_idx, out, mask=store_mask)


def conv3x3_mish_bn(x, weight, bias, scale, shift):
    N, IC, H, W = x.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # Pre-pack weight as [KH*KW*IC, OC] (K-major, N contiguous)
    # weight is [OC, IC, KH, KW]; we want w[kh, kw, ic, oc]
    w_r = weight.permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC)

    out = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

    N_HW = N * OH * OW

    grid = lambda META: (
        triton.cdiv(N_HW, META['BLOCK_M']),
        triton.cdiv(OC, META['BLOCK_N']),
    )

    _conv3x3_mish_bn_kernel[grid](
        x_nhwc, w_r, bias, scale, shift, out,
        N, H, W, IC, OH, OW, OC,
        N_HW,
    )

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