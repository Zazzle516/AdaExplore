import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'H_OUT', 'W_OUT', 'KH', 'KW'],
)
@triton.jit
def conv_hardswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_xn, stride_xh, stride_xw, stride_xc,  # NHWC
    stride_on, stride_oh, stride_ow, stride_oc,  # NHWC
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_k = tl.arange(0, BLOCK_K)

    mask_oc = offs_oc < OC
    mask_hw = offs_hw < (H_OUT * W_OUT)

    oh = offs_hw // W_OUT
    ow = offs_hw % W_OUT

    K = IC * KH * KW
    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    # weight layout: (OC, KH, KW, IC) contiguous, so stride along K = 1, stride along OC = K
    # x layout: NHWC, so for fixed (n, ih, iw), channel stride is stride_xc=1
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        mask_k = k_idx < K

        # decompose k_idx into (kh, kw, ic)
        kh = k_idx // (KW * IC)
        rem = k_idx % (KW * IC)
        kw = rem // IC
        ic = rem % IC

        # Load weight tile [BLOCK_OC, BLOCK_K]: w[oc, kh, kw, ic]
        w_off = offs_oc[:, None] * K + k_idx[None, :]
        w_mask = mask_oc[:, None] & mask_k[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        # Load input tile [BLOCK_HW, BLOCK_K]: x[n, oh+kh, ow+kw, ic]
        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw[None, :]
        x_off = pid_n * stride_xn + ih * stride_xh + iw * stride_xw + ic[None, :] * stride_xc
        x_mask = mask_hw[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        # acc[BLOCK_HW, BLOCK_OC] += x_tile[BLOCK_HW, BLOCK_K] @ w_tile.T[BLOCK_K, BLOCK_OC]
        acc += tl.dot(x_tile, tl.trans(w_tile))

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += bias[None, :]

    # fused hardswish then relu: max(0, x * clamp(x+3, 0, 6)/6)
    x_plus_3 = acc + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    hs = acc * relu6 * (1.0 / 6.0)
    out = tl.maximum(hs, 0.0)

    # store NHWC: out[n, oh, ow, oc]
    out_off = (pid_n * stride_on
               + oh[:, None] * stride_oh
               + ow[:, None] * stride_ow
               + offs_oc[None, :] * stride_oc)
    mask = mask_hw[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_off, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to (OC, KH, KW, IC) contiguous
        with torch.no_grad():
            w = self.conv.weight.detach().cuda()  # (OC, IC, KH, KW)
            w_perm = w.permute(0, 2, 3, 1).contiguous()  # (OC, KH, KW, IC)
            self.register_buffer('weight_perm', w_perm)
            self.register_buffer('bias_buf', self.conv.bias.detach().cuda().contiguous())

        OC, KH, KW = w.shape[0], w.shape[2], w.shape[3]
        K = in_channels * KH * KW
        # pick BLOCK_K as next power of 2 >= K, but at least 16
        bk = 16
        while bk < K:
            bk *= 2
        self.BLOCK_K = bk

    def forward(self, x):
        x = x.cuda()
        # to NHWC contiguous
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        N, H, W, IC = x_nhwc.shape
        OC, KH, KW, _ = self.weight_perm.shape
        H_OUT = H - KH + 1
        W_OUT = W - KW + 1

        out_nhwc = torch.empty((N, H_OUT, W_OUT, OC), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(H_OUT * W_OUT, meta['BLOCK_HW']),
        )

        conv_hardswish_relu_kernel[grid](
            x_nhwc, self.weight_perm, self.bias_buf, out_nhwc,
            N, IC, H, W,
            OC, H_OUT, W_OUT,
            KH, KW,
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            out_nhwc.stride(0), out_nhwc.stride(1), out_nhwc.stride(2), out_nhwc.stride(3),
            BLOCK_K=self.BLOCK_K,
        )

        # back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out