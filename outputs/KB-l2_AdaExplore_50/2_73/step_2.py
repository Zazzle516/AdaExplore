import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _conv_configs():
    return [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ]


@triton.autotune(configs=_conv_configs(), key=['N', 'OC', 'OH', 'OW', 'IC', 'KH', 'KW'])
@triton.jit
def conv2d_bn_scale_kernel(
    x_ptr, w_ptr, bias_ptr,
    out_ptr,
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
    pid_n = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # OC tile
    pid_s = tl.program_id(2)  # spatial tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial

    OHOW = OH * OW
    oh = offs_s // OW
    ow = offs_s % OW

    K = IC * KH * KW
    KHW = KH * KW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # hoisted base address for input
    x_base = x_ptr + pid_n * stride_xn + oh * stride_xh + ow * stride_xw  # [BLOCK_N]
    s_mask = offs_s < OHOW

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        ic = offs_k // KHW
        kh_kw = offs_k % KHW
        kh = kh_kw // KW
        kw = kh_kw % KW

        # per-k offset into input (channel + kernel spatial)
        k_off = ic * stride_xc + kh * stride_xh + kw * stride_xw  # [BLOCK_K]

        # weight load [BLOCK_M, BLOCK_K]
        w_ptrs = (w_ptr
                  + offs_m[:, None] * stride_wo
                  + ic[None, :] * stride_wi
                  + kh[None, :] * stride_wh
                  + kw[None, :] * stride_ww)
        w_mask = (offs_m[:, None] < OC) & k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # input load [BLOCK_K, BLOCK_N]
        x_ptrs = x_base[None, :] + k_off[:, None]
        x_mask = k_mask[:, None] & s_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(w_vals, x_vals)

    # apply fused bias+scale (bias_ptr contains pre-fused scale * (bn_w/sigma * (conv_b - mean) + bn_b))
    # and scale_ptr is folded into weight at host. Here we just add bias.
    bias = tl.load(bias_ptr + offs_m, mask=offs_m < OC, other=0.0)
    acc = acc + bias[:, None]

    out_ptrs = (out_ptr
                + pid_n * stride_on
                + offs_m[:, None] * stride_oc
                + oh[None, :] * stride_oh
                + ow[None, :] * stride_ow)
    out_mask = (offs_m[:, None] < OC) & (offs_s[None, :] < OH * OW)
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def _get_fused(self):
        # Fold BN + scale into conv weight/bias for eval
        w = self.conv.weight  # [OC, IC, KH, KW]
        b = self.conv.bias    # [OC]
        bn_w = self.bn.weight
        bn_b = self.bn.bias
        mean = self.bn.running_mean
        var = self.bn.running_var
        eps = self.bn.eps
        scale = self.scaling_factor

        inv_std = torch.rsqrt(var + eps)
        coef = bn_w * inv_std * scale  # [OC]
        fused_w = w * coef.view(-1, 1, 1, 1)
        fused_b = (b - mean) * coef + bn_b * scale
        return fused_w.contiguous(), fused_b.contiguous()

    def forward(self, x):
        x = x.cuda().contiguous()
        if self.training:
            # fallback path
            x = self.conv(x)
            x = self.bn(x)
            x = x * self.scaling_factor
            return x

        fused_w, fused_b = self._get_fused()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(OH * OW, meta['BLOCK_N']),
        )

        conv2d_bn_scale_kernel[grid](
            x, fused_w, fused_b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            fused_w.stride(0), fused_w.stride(1), fused_w.stride(2), fused_w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out