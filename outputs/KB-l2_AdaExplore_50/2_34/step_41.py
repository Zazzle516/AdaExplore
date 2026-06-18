import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def _convtr3d_ln_gelu_scale_kernel(
    x_ptr,           # (N, IC, D, H, W)
    w_ptr,           # (IC, OC, KD, KH, KW)
    bias_ptr,        # (OC,)
    gamma_ptr,       # (OC,)
    beta_ptr,        # (OC,)
    out_ptr,         # (N, OC, OD, OH, OW)
    N, IC, OC,
    D, H, W,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    eps, scaling_factor,
    BLOCK_OC: tl.constexpr,
):
    # program per (n, od, oh, ow); compute all OC at once
    pid = tl.program_id(0)
    # decode pid -> n, od, oh, ow
    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    oc_idx = tl.arange(0, BLOCK_OC)
    oc_mask = oc_idx < OC

    # bias
    bias = tl.load(bias_ptr + oc_idx, mask=oc_mask, other=0.0).to(tl.float32)
    acc = bias

    # For conv_transpose3d:
    # out[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw} x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where id*SD = od + PD - kd, etc. So id = (od + PD - kd) / SD if divisible, in [0, D).

    # Pre-compute (od + PD), (oh + PH), (ow + PW)
    od_p = od + PD
    oh_p = oh + PH
    ow_p = ow + PW

    for kd in tl.static_range(0, KD):
        id_num = od_p - kd
        id_ok = (id_num % SD) == 0
        id_val = id_num // SD
        id_ok = id_ok & (id_val >= 0) & (id_val < D)
        for kh in tl.static_range(0, KH):
            ih_num = oh_p - kh
            ih_ok = (ih_num % SH) == 0
            ih_val = ih_num // SH
            ih_ok = ih_ok & (ih_val >= 0) & (ih_val < H)
            for kw in tl.static_range(0, KW):
                iw_num = ow_p - kw
                iw_ok = (iw_num % SW) == 0
                iw_val = iw_num // SW
                iw_ok = iw_ok & (iw_val >= 0) & (iw_val < W)

                valid = id_ok & ih_ok & iw_ok
                if valid:
                    # gather x[n, :, id_val, ih_val, iw_val] and w[:, :, kd, kh, kw]
                    # accumulate over IC
                    # x base: n*IC*D*H*W + ic*D*H*W + id*H*W + ih*W + iw
                    x_base = n * IC * D * H * W + id_val * H * W + ih_val * W + iw_val
                    # w base: ic*OC*KD*KH*KW + oc*KD*KH*KW + kd*KH*KW + kh*KW + kw
                    w_kbase = kd * KH * KW + kh * KW + kw

                    for ic in range(0, IC):
                        x_val = tl.load(x_ptr + x_base + ic * D * H * W).to(tl.float32)
                        w_vals = tl.load(
                            w_ptr + ic * OC * KD * KH * KW + oc_idx * KD * KH * KW + w_kbase,
                            mask=oc_mask, other=0.0,
                        ).to(tl.float32)
                        acc += x_val * w_vals

    # LayerNorm across OC
    acc_masked = tl.where(oc_mask, acc, 0.0)
    mean = tl.sum(acc_masked, axis=0) / OC
    xc = tl.where(oc_mask, acc - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / OC
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(gamma_ptr + oc_idx, mask=oc_mask, other=0.0).to(tl.float32)
    b = tl.load(beta_ptr + oc_idx, mask=oc_mask, other=0.0).to(tl.float32)

    y = xc * rstd * g + b
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    out = gelu * scaling_factor

    # store with channels-last-ish layout? We want (N, OC, OD, OH, OW)
    # out index: n*OC*OD*OH*OW + oc*OD*OH*OW + od*OH*OW + oh*OW + ow
    spatial = OD * OH * OW
    out_off = n * OC * spatial + oc_idx * spatial + od * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_off, out, mask=oc_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.eps = eps
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Fallback for any unexpected shape: use reference path
        N, IC, D, H, W = x.shape
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding
        OD = (D - 1) * SD - 2 * PD + KD
        OH = (H - 1) * SH - 2 * PH + KH
        OW = (W - 1) * SW - 2 * PW + KW
        OC = self.out_channels

        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        bias = self.conv_transpose.bias
        if bias is None:
            bias = torch.zeros(OC, device=x.device, dtype=x.dtype)
        bias = bias.contiguous()

        gamma = self.layer_norm.weight.contiguous()
        beta = self.layer_norm.bias.contiguous()

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = triton.next_power_of_2(OC)
        total = N * OD * OH * OW
        grid = (total,)

        _convtr3d_ln_gelu_scale_kernel[grid](
            x, w, bias, gamma, beta, out,
            N, IC, OC,
            D, H, W,
            OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            self.eps, self.scaling_factor,
            BLOCK_OC=BLOCK_OC,
            num_warps=2,
            num_stages=2,
        )
        return out