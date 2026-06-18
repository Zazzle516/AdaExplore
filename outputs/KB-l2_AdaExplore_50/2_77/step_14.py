import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _convt3d_bn_gap_kernel(
    x_ptr,           # input (N, IC, ID, IH, IW)
    w_ptr,           # weight (IC, OC, KD, KH, KW)
    fused_bias_ptr,  # (OC,) bias term added per output element
    inv_S_ptr,       # scalar 1/S
    gap_scale_ptr,   # (OC,) = scale (sf * invstd * gamma)
    out_ptr,         # (N, OC)
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    S,
    BLOCK_IC: tl.constexpr,
):
    # one program per (n, oc)
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    # accumulator: sum over output spatial of conv result
    # output[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw} x[n, ic, od-kd, oh-kh, ow-kw] * w[ic, oc, kd, kh, kw]
    # for ConvTranspose3d (stride=1, padding=0).
    # We want sum over (od, oh, ow) of output[n, oc, od, oh, ow].
    # Swap order: sum_{ic, kd, kh, kw} w[ic,oc,kd,kh,kw] * sum_{id, ih, iw} x[n,ic,id,ih,iw] * count(...)
    # But here every (id, ih, iw, kd, kh, kw) maps to exactly one (od, oh, ow), so sum over output =
    # sum over (id, ih, iw, ic, kd, kh, kw) of x[n,ic,id,ih,iw] * w[ic,oc,kd,kh,kw]
    # = sum_ic (sum_{id,ih,iw} x) * (sum_{kd,kh,kw} w)
    # because for each input voxel there's a kernel-shaped output region all getting added, but summing all
    # contributions sums to (sum_x_per_ic) * (sum_w_per_ic_oc).

    # So we just need: per ic: x_sum[n, ic] * w_sum[ic, oc], accumulated, then scaled.

    acc = 0.0
    spatial_in = ID * IH * IW
    kvol = KD * KH * KW

    for ic_start in range(0, IC, BLOCK_IC):
        ic_offs = ic_start + tl.arange(0, BLOCK_IC)
        ic_mask = ic_offs < IC

        # compute x_sum[n, ic] for each ic in block
        # need to reduce x[n, ic, :, :, :]
        # We do it by ic in a small loop
        # Actually let's just loop scalar over ic for simplicity since IC is small (64)
        pass

    # simpler: scalar loop over ic
    acc = 0.0
    for ic in range(0, IC):
        # x_sum over spatial
        x_base = (n * IC + ic) * spatial_in
        xs = 0.0
        # spatial_in = 16*32*32 = 16384
        BLOCK_X: tl.constexpr = 2048
        for off in range(0, spatial_in, BLOCK_X):
            idx = off + tl.arange(0, BLOCK_X)
            m = idx < spatial_in
            v = tl.load(x_ptr + x_base + idx, mask=m, other=0.0)
            xs += tl.sum(v)

        # w_sum over kernel for (ic, oc)
        w_base = ((ic * OC) + oc) * kvol
        ws = 0.0
        BLOCK_W: tl.constexpr = 128
        for off in range(0, kvol, BLOCK_W):
            idx = off + tl.arange(0, BLOCK_W)
            m = idx < kvol
            v = tl.load(w_ptr + w_base + idx, mask=m, other=0.0)
            ws += tl.sum(v)

        acc += xs * ws

    # acc = sum over output spatial of conv(x)[n,oc,:,:,:] (without conv_bias)
    # Need to add conv_bias contribution: conv_bias[oc] * (OD*OH*OW)
    # That's already folded into fused_bias term as bias * S? Let's handle outside.

    # Apply scale and BN: gap = (1/S) * sum( (conv_out * sf - rm) * invstd * gamma + beta )
    #                       = scale * (1/S) * sum(conv_out) + fused_bias_const
    # where scale = sf * invstd * gamma, fused_bias_const = beta - rm * invstd * gamma
    # But conv_out includes conv_bias[oc], so sum(conv_out) = acc + conv_bias[oc] * S
    # We'll incorporate conv_bias into fused_bias_ptr: it stores beta - rm*invstd*gamma + conv_bias*sf*invstd*gamma
    # Wait: scale*(conv_bias) + fused_bias_const. So fused_bias_ptr stores
    #    final_bias = beta - rm*invstd*gamma + conv_bias * sf * invstd * gamma
    # Then result = scale * (acc / S) + final_bias

    inv_S = tl.load(inv_S_ptr)
    s = tl.load(gap_scale_ptr + oc)
    fb = tl.load(fused_bias_ptr + oc)

    result = acc * inv_S * s + fb
    tl.store(out_ptr + n * OC + oc, result)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = float(scale_factor)
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        if self.training:
            x = self.conv_transpose(x)
            x = x * self.scale_factor
            x = self.batch_norm(x)
            x = self.global_avg_pool(x)
            return x

        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OD = ID + KD - 1
        OH = IH + KH - 1
        OW = IW + KW - 1
        S = OD * OH * OW
        OC = self.out_channels

        # BN folded params
        rm = self.batch_norm.running_mean
        rv = self.batch_norm.running_var
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        eps = self.batch_norm.eps

        invstd = torch.rsqrt(rv + eps)
        gap_scale = self.scale_factor * invstd * gamma  # (OC,)
        conv_bias = self.conv_transpose.bias  # (OC,)
        fused_bias = beta - rm * invstd * gamma + conv_bias * gap_scale  # (OC,)

        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)

        out = torch.empty((N, OC), device=x.device, dtype=x.dtype)
        inv_S = torch.tensor(1.0 / S, device=x.device, dtype=x.dtype)

        grid = (N * OC,)
        _convt3d_bn_gap_kernel[grid](
            x, weight, fused_bias, inv_S, gap_scale, out,
            N, IC, OC,
            ID, IH, IW,
            OD, OH, OW,
            KD, KH, KW,
            S,
            BLOCK_IC=16,
            num_warps=4,
        )
        return out.view(N, OC, 1, 1, 1)