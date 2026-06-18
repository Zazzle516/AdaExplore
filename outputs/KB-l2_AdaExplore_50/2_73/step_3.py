import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'K_FLAT'],
)
@triton.jit
def conv_bn_scale_kernel(
    x_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    OUT_HW, K_FLAT,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program_id(0): batch index
    # program_id(1): OC tile
    # program_id(2): spatial tile
    n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_sp = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_oc = offs_oc < OC
    mask_sp = offs_sp < OUT_HW

    # decode spatial -> oh, ow
    oh = offs_sp // OW
    ow = offs_sp % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K_FLAT = IC*KH*KW
    for k_start in range(0, K_FLAT, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = offs_k < K_FLAT

        # decompose k -> ic, kh, kw
        ic = offs_k // (KH * KW)
        rem = offs_k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # weight load: w[oc, ic, kh, kw] -> shape [BLOCK_M, BLOCK_K]
        w_offsets = (offs_oc[:, None] * (IC * KH * KW)
                     + ic[None, :] * (KH * KW)
                     + kh[None, :] * KW
                     + kw[None, :])
        w_mask = mask_oc[:, None] & mask_k[None, :]
        w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        # input load: x[n, ic, oh+kh, ow+kw] -> shape [BLOCK_K, BLOCK_N]
        ih = oh[None, :] + kh[:, None]  # [BLOCK_K, BLOCK_N]
        iw = ow[None, :] + kw[:, None]
        ic_b = ic[:, None]  # [BLOCK_K, 1]

        x_offsets = (n * (IC * IH * IW)
                     + ic_b * (IH * IW)
                     + ih * IW
                     + iw)
        x_mask = (mask_k[:, None] & mask_sp[None, :]
                  & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW))
        x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        acc += tl.dot(w_vals, x_vals)

    # epilogue: scale (folded bn*scaling) and bias
    scale = tl.load(scale_ptr + offs_oc, mask=mask_oc, other=0.0)
    bias = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)

    out = acc * scale[:, None] + bias[:, None]

    out_offsets = (n * (OC * OUT_HW)
                   + offs_oc[:, None] * OUT_HW
                   + offs_sp[None, :])
    out_mask = mask_oc[:, None] & mask_sp[None, :]
    tl.store(out_ptr + out_offsets, out, mask=out_mask)


def conv_bn_scale(x, weight, scale, bias):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1
    OUT_HW = OH * OW
    K_FLAT = IC * KH * KW

    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        N,
        triton.cdiv(OC, meta['BLOCK_M']),
        triton.cdiv(OUT_HW, meta['BLOCK_N']),
    )

    conv_bn_scale_kernel[grid](
        x, weight, scale, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        OUT_HW, K_FLAT,
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

        # Eval mode: fold BN + scaling into per-OC scale/bias
        w = self.conv.weight
        cb = self.conv.bias
        rm = self.bn.running_mean
        rv = self.bn.running_var
        eps = self.bn.eps
        bw = self.bn.weight
        bb = self.bn.bias

        invstd = torch.rsqrt(rv + eps)
        scale = bw * invstd * self.scaling_factor  # [OC]
        if cb is not None:
            bias = (cb - rm) * bw * invstd * self.scaling_factor + bb * self.scaling_factor
        else:
            bias = (-rm) * bw * invstd * self.scaling_factor + bb * self.scaling_factor

        scale = scale.contiguous()
        bias = bias.contiguous()

        return conv_bn_scale(x, w, scale, bias)