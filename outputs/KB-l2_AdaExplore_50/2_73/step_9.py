import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
    ],
    key=['N', 'OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_bn_scale_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,  # full OC (64)
    BLOCK_N: tl.constexpr,  # spatial tile
    BLOCK_K: tl.constexpr,  # IC*KH*KW (72) padded
    K_REAL: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_s = tl.program_id(1)  # spatial tile

    offs_m = tl.arange(0, BLOCK_M)            # OC
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    k_mask = offs_k < K_REAL
    s_mask = offs_s < (OH * OW)

    oh = offs_s // OW
    ow = offs_s % OW

    ic = offs_k // (KH * KW)
    kh_kw = offs_k % (KH * KW)
    kh = kh_kw // KW
    kw = kh_kw % KW

    # Load weight tile [BLOCK_M, BLOCK_K] - register resident
    w_ptrs = (w_ptr
              + offs_m[:, None] * stride_wo
              + ic[None, :] * stride_wi
              + kh[None, :] * stride_wh
              + kw[None, :] * stride_ww)
    w_mask = (offs_m[:, None] < OC) & k_mask[None, :]
    w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

    # Load input tile [BLOCK_K, BLOCK_N]
    ih = oh[None, :] + kh[:, None]
    iw = ow[None, :] + kw[:, None]
    x_ptrs = (x_ptr
              + pid_n * stride_xn
              + ic[:, None] * stride_xc
              + ih * stride_xh
              + iw * stride_xw)
    x_mask = k_mask[:, None] & s_mask[None, :]
    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

    acc = tl.dot(w_vals, x_vals)

    bias = tl.load(bias_ptr + offs_m, mask=offs_m < OC, other=0.0)
    acc = acc + bias[:, None]

    out_ptrs = (out_ptr
                + pid_n * stride_on
                + offs_m[:, None] * stride_oc
                + oh[None, :] * stride_oh
                + ow[None, :] * stride_ow)
    out_mask = (offs_m[:, None] < OC) & s_mask[None, :]
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
        self._cache = None

    def _get_fused(self):
        w = self.conv.weight
        b = self.conv.bias
        bn_w = self.bn.weight
        bn_b = self.bn.bias
        mean = self.bn.running_mean
        var = self.bn.running_var
        eps = self.bn.eps
        scale = self.scaling_factor

        inv_std = torch.rsqrt(var + eps)
        coef = bn_w * inv_std * scale
        fused_w = w * coef.view(-1, 1, 1, 1)
        fused_b = (b - mean) * coef + bn_b * scale
        return fused_w.contiguous(), fused_b.contiguous()

    def forward(self, x):
        x = x.cuda().contiguous()
        if self.training:
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

        # OC=64 fixed, K_REAL = IC*KH*KW = 8*3*3 = 72; pad to 128
        BLOCK_M = 64
        K_REAL = IC * KH * KW
        # Pad K to next power-of-2 / multiple acceptable by tl.dot (min 16)
        if K_REAL <= 16:
            BLOCK_K = 16
        elif K_REAL <= 32:
            BLOCK_K = 32
        elif K_REAL <= 64:
            BLOCK_K = 64
        elif K_REAL <= 128:
            BLOCK_K = 128
        else:
            BLOCK_K = triton.next_power_of_2(K_REAL)

        # Ensure OC matches BLOCK_M else pad up
        if OC > BLOCK_M:
            BLOCK_M = triton.next_power_of_2(OC)

        grid = lambda meta: (
            N,
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
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            K_REAL=K_REAL,
        )
        return out