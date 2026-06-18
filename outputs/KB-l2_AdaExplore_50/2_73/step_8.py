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
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_bn_scale_kernel(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    OC_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # program_id(0): batch
    # program_id(1): oh row index
    # program_id(2): ow tile
    n = tl.program_id(0)
    oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    offs_ow = pid_ow * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_ow = offs_ow < OW

    offs_oc = tl.arange(0, OC_C)  # [OC_C]
    mask_oc = offs_oc < OC

    acc = tl.zeros((OC_C, BLOCK_N), dtype=tl.float32)

    # K_FLAT = IC * KH * KW, fully unrolled across KH, KW; inner loop over IC chunk
    # but here IC is small (8), so a single IC tile suffices
    offs_ic = tl.arange(0, IC_C)  # [IC_C]
    mask_ic = offs_ic < IC

    for kh in tl.static_range(0, KH):
        ih = oh + kh
        for kw in tl.static_range(0, KW):
            iw = offs_ow + kw  # [BLOCK_N]
            # weight: w[oc, ic, kh, kw], shape [OC_C, IC_C]
            w_off = (offs_oc[:, None] * (IC * KH * KW)
                     + offs_ic[None, :] * (KH * KW)
                     + kh * KW + kw)
            w_mask = mask_oc[:, None] & mask_ic[None, :]
            w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [OC_C, IC_C]

            # input: x[n, ic, ih, iw], shape [IC_C, BLOCK_N]
            x_off = (n * (IC * IH * IW)
                     + offs_ic[:, None] * (IH * IW)
                     + ih * IW
                     + iw[None, :])
            x_mask = mask_ic[:, None] & mask_ow[None, :]
            x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [IC_C, BLOCK_N]

            acc += tl.dot(w_vals, x_vals)

    scale = tl.load(scale_ptr + offs_oc, mask=mask_oc, other=0.0)
    bias = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    out = acc * scale[:, None] + bias[:, None]

    out_off = (n * (OC * OH * OW)
               + offs_oc[:, None] * (OH * OW)
               + oh * OW
               + offs_ow[None, :])
    out_mask = mask_oc[:, None] & mask_ow[None, :]
    tl.store(out_ptr + out_off, out, mask=out_mask)


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

    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    IC_C = max(16, _next_pow2(IC))
    OC_C = _next_pow2(OC)

    grid = lambda meta: (
        N,
        OH,
        triton.cdiv(OW, meta['BLOCK_N']),
    )

    conv_bn_scale_kernel[grid](
        x, weight, scale, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        IC_C, OC_C,
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

        scale = scale.contiguous()
        bias = bias.contiguous()

        return conv_bn_scale(x, w, scale, bias)