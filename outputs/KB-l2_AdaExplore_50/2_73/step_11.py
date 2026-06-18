import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_bn_scale_kernel(
    x_ptr, w_ptr, scale_ptr, shift_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile (OH*OW)
    BLOCK_K: tl.constexpr,  # IC*KH*KW reduction tile
):
    pid_n = tl.program_id(0)  # batch
    pid_m = tl.program_id(1)  # OC tile
    pid_s = tl.program_id(2)  # spatial tile

    offs_oc = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_sp = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)

    oh = offs_sp // OW
    ow = offs_sp % OW

    K = IC * KH * KW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k  # [BLOCK_K]
        k_mask = k < K

        ic = k // (KH * KW)
        rem = k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # weight: [BLOCK_M, BLOCK_K]
        w_ptrs = w_ptr + offs_oc[:, None] * stride_wo + ic[None, :] * stride_wi + kh[None, :] * stride_wh + kw[None, :] * stride_ww
        w_mask = (offs_oc[:, None] < OC) & k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # input: [BLOCK_K, BLOCK_N]
        ih = oh[None, :] + kh[:, None]  # [BLOCK_K, BLOCK_N]
        iw = ow[None, :] + kw[:, None]
        in_bounds = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & k_mask[:, None] & (offs_sp[None, :] < OH * OW)
        x_ptrs = x_ptr + pid_n * stride_xn + ic[:, None] * stride_xc + ih * stride_xh + iw * stride_xw
        x_vals = tl.load(x_ptrs, mask=in_bounds, other=0.0)

        acc += tl.dot(w_vals, x_vals)

    # Apply scale and shift (BN + scaling folded)
    scale = tl.load(scale_ptr + offs_oc, mask=offs_oc < OC, other=0.0)
    shift = tl.load(shift_ptr + offs_oc, mask=offs_oc < OC, other=0.0)
    out = acc * scale[:, None] + shift[:, None]

    out_ptrs = out_ptr + pid_n * stride_on + offs_oc[:, None] * stride_oc + oh[None, :] * stride_oh + ow[None, :] * stride_ow
    out_mask = (offs_oc[:, None] < OC) & (offs_sp[None, :] < OH * OW)
    tl.store(out_ptrs, out, mask=out_mask)


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
            # fallback
            y = self.conv(x)
            y = self.bn(y)
            y = y * self.scaling_factor
            return y

        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        weight = self.conv.weight  # [OC, IC, KH, KW]
        conv_bias = self.conv.bias  # [OC] or None

        bn_w = self.bn.weight
        bn_b = self.bn.bias
        bn_mean = self.bn.running_mean
        bn_var = self.bn.running_var
        bn_eps = self.bn.eps

        inv = torch.rsqrt(bn_var + bn_eps)
        scale = (bn_w * inv) * self.scaling_factor  # [OC]
        if conv_bias is not None:
            shift = ((conv_bias - bn_mean) * inv * bn_w + bn_b) * self.scaling_factor
        else:
            shift = ((-bn_mean) * inv * bn_w + bn_b) * self.scaling_factor

        scale = scale.contiguous()
        shift = shift.contiguous()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        S = OH * OW
        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(S, meta['BLOCK_N']),
        )

        conv_bn_scale_kernel[grid](
            x, weight, scale, shift, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out