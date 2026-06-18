import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'H_OUT', 'W_OUT', 'KH', 'KW', 'N'],
)
@triton.jit
def conv_hardswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, H, W,
    OC, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    K: tl.constexpr,  # IC * KH * KW
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    mask_oc = offs_oc < OC
    mask_hw = offs_hw < (H_OUT * W_OUT)

    oh = offs_hw // W_OUT
    ow = offs_hw % W_OUT

    # Load weight tile [BLOCK_OC, K] once
    offs_k = tl.arange(0, K)
    # decompose k = ic*KH*KW + kh*KW + kw
    k_kw = offs_k % KW
    k_kh = (offs_k // KW) % KH
    k_ic = offs_k // (KH * KW)

    # weight: [OC, IC, KH, KW] contiguous => offset = oc*(IC*KH*KW) + ic*KH*KW + kh*KW + kw = oc*K + k
    w_off = offs_oc[:, None] * K + offs_k[None, :]
    w_mask = mask_oc[:, None] & (offs_k[None, :] < K)
    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_OC, K]

    # Build input gather: x[n, ic, oh+kh, ow+kw]
    # x stride: ic stride = H*W, h stride = W, w stride = 1, n stride = IC*H*W
    # Build x offsets [K, BLOCK_HW]
    ih = oh[None, :] + k_kh[:, None]  # [K, BLOCK_HW]
    iw = ow[None, :] + k_kw[:, None]
    x_base = pid_n * (IC * H * W) + k_ic[:, None] * (H * W) + ih * W + iw
    x_mask = mask_hw[None, :]  # offs_k always < K since K is constexpr matching tl.arange

    x_tile = tl.load(x_ptr + x_base, mask=x_mask, other=0.0)  # [K, BLOCK_HW]

    acc = tl.dot(w_tile, x_tile, out_dtype=tl.float32)  # [BLOCK_OC, BLOCK_HW]

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += bias[:, None]

    x_plus_3 = acc + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    hs = acc * relu6 * (1.0 / 6.0)
    out = tl.maximum(hs, 0.0)

    # output: [N, OC, H_OUT, W_OUT] contiguous
    out_off = (pid_n * (OC * H_OUT * W_OUT)
               + offs_oc[:, None] * (H_OUT * W_OUT)
               + (oh[None, :] * W_OUT + ow[None, :]))
    mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_off, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        N, IC, H, W = x.shape
        OC, _, KH, KW = weight.shape
        H_OUT = H - KH + 1
        W_OUT = W - KW + 1
        K = IC * KH * KW

        out = torch.empty((N, OC, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(H_OUT * W_OUT, meta['BLOCK_HW']),
        )

        conv_hardswish_relu_kernel[grid](
            x, weight, bias, out,
            N, IC, H, W,
            OC, H_OUT, W_OUT,
            KH, KW, K,
        )
        return out