import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 32},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'K_FLAT'],
)
@triton.jit
def conv_bn_scale_kernel(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    K_FLAT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program_id(0): batch index
    # program_id(1): spatial tile over OH*OW (flattened)
    n = tl.program_id(0)
    pid_s = tl.program_id(1)

    OHOW = OH * OW

    offs_oc = tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_oc = offs_oc < OC
    mask_s = offs_s < OHOW

    oh = offs_s // OW
    ow = offs_s - oh * OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Load scale & bias for OC tile
    scale = tl.load(scale_ptr + offs_oc, mask=mask_oc, other=0.0)
    bias = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)

    # K-loop: tile K_FLAT with BLOCK_K
    KHW = KH * KW
    for kk in tl.static_range(0, tl.cdiv(K_FLAT, BLOCK_K)):
        offs_k = kk * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_FLAT

        ic_k = offs_k // KHW
        rem_k = offs_k - ic_k * KHW
        kh_k = rem_k // KW
        kw_k = rem_k - kh_k * KW

        # Weight tile: [BLOCK_M, BLOCK_K]
        w_offsets = (offs_oc[:, None] * (IC * KHW)
                     + ic_k[None, :] * KHW
                     + kh_k[None, :] * KW
                     + kw_k[None, :])
        w_mask = mask_oc[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        # Input tile: [BLOCK_K, BLOCK_N]
        ih = oh[None, :] + kh_k[:, None]  # [BLOCK_K, BLOCK_N]
        iw = ow[None, :] + kw_k[:, None]  # [BLOCK_K, BLOCK_N]

        x_offsets = (n * (IC * IH * IW)
                     + ic_k[:, None] * (IH * IW)
                     + ih * IW
                     + iw)
        x_mask = mask_s[None, :] & k_mask[:, None]
        x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        acc += tl.dot(w_vals, x_vals)

    out = acc * scale[:, None] + bias[:, None]

    out_offsets = (n * (OC * OHOW)
                   + offs_oc[:, None] * OHOW
                   + offs_s[None, :])
    out_mask = mask_oc[:, None] & mask_s[None, :]
    tl.store(out_ptr + out_offsets, out, mask=out_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def conv_bn_scale(x, weight, scale, bias):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1
    K_FLAT = IC * KH * KW

    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_M = _next_pow2(OC)
    BLOCK_K = 16

    grid = lambda meta: (
        N,
        triton.cdiv(OH * OW, meta['BLOCK_N']),
    )

    conv_bn_scale_kernel[grid](
        x, weight, scale, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        K_FLAT,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.training:
            x = self.conv(x)
            x = self.bn(x)
            x = x * self.scaling_factor
            return x

        w = self.conv.weight
        cb = self.conv.bias
        rm = self.bn.running_mean
        rv = self.bn.running_var
        eps = self.bn.eps
        bw = self.bn.weight
        bb = self.bn.bias

        invstd = torch.rsqrt(rv + eps)
        scale = bw * invstd * self.scaling_factor
        if cb is not None:
            bias = (cb - rm) * bw * invstd * self.scaling_factor + bb * self.scaling_factor
        else:
            bias = (-rm) * bw * invstd * self.scaling_factor + bb * self.scaling_factor

        OC, IC, KH, KW = w.shape
        OC_P = _next_pow2(OC)

        if OC_P != OC:
            w_new = torch.zeros((OC_P, IC, KH, KW), device=w.device, dtype=w.dtype)
            w_new[:OC, :, :, :] = w
            w = w_new
            scale_new = torch.zeros((OC_P,), device=scale.device, dtype=scale.dtype)
            bias_new = torch.zeros((OC_P,), device=bias.device, dtype=bias.dtype)
            scale_new[:OC] = scale
            bias_new[:OC] = bias
            scale = scale_new
            bias = bias_new

        scale = scale.contiguous()
        bias = bias.contiguous()
        out = conv_bn_scale(x, w, scale, bias)
        if out.shape[1] != OC:
            out = out[:, :OC, :, :].contiguous()
        return out