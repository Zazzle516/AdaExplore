import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _conv_configs():
    cfgs = []
    for bm in [32, 64, 128]:
        for bn in [32, 64, 128]:
            for bk in [16, 32, 64]:
                for nw in [4, 8]:
                    for ns in [2, 3]:
                        smem = (bm * bk + bk * bn) * 4 * ns
                        if smem > 96 * 1024:
                            continue
                        if bm * bn > 128 * 128:
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
    K_PER_KHKW = IC  # channel count per (kh,kw) block
    KHKW_IC = 9 * IC  # KH=KW=3

    # explicit unroll over (kh, kw) = 9 iterations; inner loop only over IC.
    for kh in tl.static_range(0, 3):
        for kw in tl.static_range(0, 3):
            # base input offset for this (kh,kw): n*HWIC + (oh+kh)*WIC + (ow+kw)*IC
            base_x = n_idx * HWIC + (oh_idx + kh) * WIC + (ow_idx + kw) * IC  # (BM,)
            # base weight offset: (kh*3 + kw)*IC + oc*KHKW_IC
            khkw_off = (kh * 3 + kw) * IC

            for ic_start in range(0, IC, BLOCK_K):
                offs_k = ic_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < IC

                x_idx = base_x[:, None] + offs_k[None, :]
                x_load_mask = m_mask[:, None] & k_mask[None, :]
                x_tile = tl.load(x_ptr + x_idx, mask=x_load_mask, other=0.0)

                # weight: (OC, KHKW_IC), index: oc * KHKW_IC + khkw_off + ic
                w_idx = offs_n[:, None] * KHKW_IC + khkw_off + offs_k[None, :]
                w_load_mask = n_mask[:, None] & k_mask[None, :]
                w_tile = tl.load(w_ptr + w_idx, mask=w_load_mask, other=0.0)

                acc += tl.dot(x_tile, tl.trans(w_tile))

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
    w_r = weight.permute(0, 2, 3, 1).contiguous().view(OC, KH * KW * IC)

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