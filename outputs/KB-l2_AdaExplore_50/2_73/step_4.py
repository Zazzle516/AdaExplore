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
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'OW', 'K_FLAT'],
)
@triton.jit
def conv_bn_scale_kernel(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    K_FLAT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # program_id(0): batch index
    # program_id(1): oh row (one program per output row)
    # program_id(2): ow tile
    n = tl.program_id(0)
    oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    offs_oc = tl.arange(0, BLOCK_M)  # [BLOCK_M], BLOCK_M == OC (padded)
    offs_ow = pid_ow * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_oc = offs_oc < OC
    mask_ow = offs_ow < OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Unrolled K loop over IC * KH * KW.
    offs_k = tl.arange(0, K_FLAT)
    ic_k = offs_k // (KH * KW)
    rem_k = offs_k % (KH * KW)
    kh_k = rem_k // KW
    kw_k = rem_k % KW

    # Weight tile: [BLOCK_M, K_FLAT]
    w_offsets = (offs_oc[:, None] * (IC * KH * KW)
                 + ic_k[None, :] * (KH * KW)
                 + kh_k[None, :] * KW
                 + kw_k[None, :])
    w_mask = mask_oc[:, None]
    w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)  # [BLOCK_M, K_FLAT]

    # Input tile: [K_FLAT, BLOCK_N]
    ih = oh + kh_k[:, None]  # [K_FLAT, 1]
    iw = offs_ow[None, :] + kw_k[:, None]  # [K_FLAT, BLOCK_N]
    ic_b = ic_k[:, None]  # [K_FLAT, 1]

    x_offsets = (n * (IC * IH * IW)
                 + ic_b * (IH * IW)
                 + ih * IW
                 + iw)
    x_mask = mask_ow[None, :] & (iw < IW)
    x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

    acc = tl.dot(w_vals, x_vals)

    scale = tl.load(scale_ptr + offs_oc, mask=mask_oc, other=0.0)
    bias = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)

    out = acc * scale[:, None] + bias[:, None]

    out_offsets = (n * (OC * OH * OW)
                   + offs_oc[:, None] * (OH * OW)
                   + oh * OW
                   + offs_ow[None, :])
    out_mask = mask_oc[:, None] & mask_ow[None, :]
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
    K_FLAT_P = _next_pow2(K_FLAT)
    # We need BLOCK_M and K_FLAT to be a power-of-2 for tl.dot; in our case OC=64, K_FLAT=72->128.
    # To keep the unrolled loop static, pad K_FLAT to the next power of two and the kernel masks via ic_k branch.

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
        K_FLAT_P,
        BLOCK_M=BLOCK_M,
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

        # Pad weight K-dim to power-of-2 if needed: but easier — pad weight to OC_padded * K_FLAT_padded.
        OC, IC, KH, KW = w.shape
        K_FLAT = IC * KH * KW
        K_FLAT_P = _next_pow2(K_FLAT)
        OC_P = _next_pow2(OC)

        if K_FLAT_P != K_FLAT or OC_P != OC:
            w_flat = w.reshape(OC, K_FLAT)
            w_pad = torch.zeros((OC_P, K_FLAT_P), device=w.device, dtype=w.dtype)
            w_pad[:OC, :K_FLAT] = w_flat
            # Reshape back: we need shape [OC_P, IC, KH, KW] but K_FLAT_P may not equal IC*KH*KW.
            # Instead, change the kernel's K loop to use K_FLAT_P with masking via (offs_k < K_FLAT).
            # Simpler: pass a padded weight that's [OC_P, K_FLAT_P], and pass IC, KH, KW unchanged.
            # The kernel uses (IC * KH * KW) as stride for OC; that must equal K_FLAT_P.
            # So we treat IC*KH*KW logically as K_FLAT_P by padding IC dimension.
            # We'll handle this by constructing a padded weight where stride matches K_FLAT_P.
            # Easiest: rebuild w to shape [OC_P, IC_P, KH, KW] where IC_P*KH*KW = K_FLAT_P.
            # But that complicates things. Instead, pad to a full [OC_P, K_FLAT_P] and have kernel treat
            # weight stride as K_FLAT_P. We override IC, KH, KW such that IC*KH*KW = K_FLAT_P.
            # Approach: pass IC_eff = K_FLAT_P // (KH*KW) rounded up; but ic decomposition would break.
            #
            # Simpler approach: pad K_FLAT only when needed by padding IC.
            # Add zero input channels until IC*KH*KW >= K_FLAT_P (which is power of 2).
            IC_NEW = K_FLAT_P // (KH * KW)
            if IC_NEW * KH * KW != K_FLAT_P:
                # K_FLAT_P not divisible — fall back
                IC_NEW = IC
                K_FLAT_P = K_FLAT
            if IC_NEW != IC or OC_P != OC:
                w_new = torch.zeros((OC_P, IC_NEW, KH, KW), device=w.device, dtype=w.dtype)
                w_new[:OC, :IC, :, :] = w
                w = w_new
                if IC_NEW != IC:
                    x_new = torch.zeros((x.shape[0], IC_NEW, x.shape[2], x.shape[3]),
                                        device=x.device, dtype=x.dtype)
                    x_new[:, :IC, :, :] = x
                    x = x_new
                if OC_P != OC:
                    scale_new = torch.zeros((OC_P,), device=scale.device, dtype=scale.dtype)
                    bias_new = torch.zeros((OC_P,), device=bias.device, dtype=bias.dtype)
                    scale_new[:OC] = scale
                    bias_new[:OC] = bias
                    scale = scale_new
                    bias = bias_new

        scale = scale.contiguous()
        bias = bias.contiguous()
        out = conv_bn_scale(x, w, scale, bias)
        # Slice back to original OC
        if out.shape[1] != OC:
            out = out[:, :OC, :, :].contiguous()
        return out