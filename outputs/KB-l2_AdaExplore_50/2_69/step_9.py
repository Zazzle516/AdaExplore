import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 512}, num_warps=8, num_stages=2),
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
    K_PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_k = tl.arange(0, K_PAD)

    mask_oc = offs_oc < OC
    mask_hw = offs_hw < (H_OUT * W_OUT)

    oh = offs_hw // W_OUT
    ow = offs_hw % W_OUT

    K = IC * KH * KW
    mask_k = offs_k < K

    # decompose k into (kh, kw, ic)
    kh = offs_k // (KW * IC)
    rem = offs_k % (KW * IC)
    kw = rem // IC
    ic = rem % IC

    # Weight layout: (K_PAD, OC) -- stored padded with zeros along K
    # Load w_tile [K_PAD, BLOCK_OC]: w[k, oc]
    w_off = offs_k[:, None] * OC + offs_oc[None, :]
    w_mask = mask_k[:, None] & mask_oc[None, :]
    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

    # Load input tile [BLOCK_HW, K_PAD]: x[n, oh+kh, ow+kw, ic]
    ih = oh[:, None] + kh[None, :]
    iw = ow[:, None] + kw[None, :]
    x_off = pid_n * stride_xn + ih * stride_xh + iw * stride_xw + ic[None, :] * stride_xc
    x_mask = mask_hw[:, None] & mask_k[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

    # acc[BLOCK_HW, BLOCK_OC] = x_tile[BLOCK_HW, K_PAD] @ w_tile[K_PAD, BLOCK_OC]
    acc = tl.dot(x_tile, w_tile)

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

        with torch.no_grad():
            w = self.conv.weight.detach().cuda()  # (OC, IC, KH, KW)
            OC, IC, KH, KW = w.shape
            K = IC * KH * KW
            # next multiple of 16 >= K, and at least 16
            K_PAD = ((K + 15) // 16) * 16
            if K_PAD < 16:
                K_PAD = 16
            # Build padded weight of shape (K_PAD, OC):
            # First permute to (KH, KW, IC, OC) then flatten to (K, OC), then pad
            w_perm = w.permute(2, 3, 1, 0).contiguous().view(K, OC)
            w_pad = torch.zeros((K_PAD, OC), dtype=w.dtype, device=w.device)
            w_pad[:K].copy_(w_perm)
            self.register_buffer('weight_pad', w_pad.contiguous())
            self.register_buffer('bias_buf', self.conv.bias.detach().cuda().contiguous())

        self.K_PAD = K_PAD
        self.OC = OC
        self.KH = KH
        self.KW = KW

    def forward(self, x):
        x = x.cuda()
        # to NHWC contiguous
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        N, H, W, IC = x_nhwc.shape
        OC = self.OC
        KH = self.KH
        KW = self.KW
        H_OUT = H - KH + 1
        W_OUT = W - KW + 1

        out_nhwc = torch.empty((N, H_OUT, W_OUT, OC), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(H_OUT * W_OUT, meta['BLOCK_HW']),
        )

        conv_hardswish_relu_kernel[grid](
            x_nhwc, self.weight_pad, self.bias_buf, out_nhwc,
            N, IC, H, W,
            OC, H_OUT, W_OUT,
            KH, KW,
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            out_nhwc.stride(0), out_nhwc.stride(1), out_nhwc.stride(2), out_nhwc.stride(3),
            K_PAD=self.K_PAD,
        )

        # back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out