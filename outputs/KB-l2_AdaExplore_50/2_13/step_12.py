import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_meanpool_kernel(
    x_ptr,        # (B, IC, D, H, W)
    w_ptr,        # (IC, OC, KD, KH, KW)
    cb_ptr,       # (OC,) conv bias
    out_ptr,      # (B, OC, H, W)  -- mean over D already divided
    B, IC, D, H, W,
    OC, KD, KH, KW,
    PAD_D, PAD_H, PAD_W,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # stride=1, output spatial = input spatial when pad = (k-1)//2 (=1, k=3)
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_h = offs_hw // W
    offs_w = offs_hw % W
    mask_hw = offs_hw < (H * W)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    inv_D = 1.0 / D

    # ConvTranspose3d with stride=1 is correlation with flipped weight.
    # y[n,oc,d,h,w] = sum_{ic,kd,kh,kw} x[n,ic, d+kd-pad, h+kh-pad, w+kw-pad] * w[ic,oc,KD-1-kd,KH-1-kh,KW-1-kw]
    # Equivalently: y[n,oc,d,h,w] = sum_{ic,kd',kh',kw'} x[n,ic, d - kd' + pad, ...] * w[ic,oc,kd',kh',kw']
    # We accumulate mean over d directly.

    for ic in range(0, IC):
        for kd in range(0, KD):
            for kh in range(0, KH):
                for kw in range(0, KW):
                    # input spatial position for output (h,w) and kernel (kh,kw)
                    # using the "flipped" formulation:
                    # in_h = h + kh - PAD_H ? Let's be precise: convT with stride 1, padding p
                    # out[h] = sum_{kh} x[h + kh - (K-1) + p]  (one common derivation)
                    # For k=3, p=1: out[h] = sum_{kh in 0..2} x[h + kh - 2 + 1] = sum x[h+kh-1]
                    # That's a standard 3x3 correlation with weight flipped.
                    in_h = offs_h + kh - (KH - 1) + PAD_H
                    in_w = offs_w + kw - (KW - 1) + PAD_W
                    mask_h = (in_h >= 0) & (in_h < H)
                    mask_w = (in_w >= 0) & (in_w < W)

                    # Sum over d: in_d = d + kd - (KD-1) + PAD_D, for d in [0,D)
                    # We need sum over d of x[in_d] valid; then divide by D.
                    # in_d range: from (kd - KD + 1 + PAD_D) to (D-1 + kd - KD + 1 + PAD_D)
                    # We'll loop over d.
                    sum_x = tl.zeros((BLOCK_HW,), dtype=tl.float32)
                    for d in range(0, D):
                        in_d = d + kd - (KD - 1) + PAD_D
                        mask_d = (in_d >= 0) & (in_d < D)
                        x_off = (((pid_b * IC + ic) * D + in_d) * H + in_h) * W + in_w
                        m = mask_hw & mask_h & mask_w & mask_d
                        xv = tl.load(x_ptr + x_off, mask=m, other=0.0)
                        sum_x += xv

                    mean_x = sum_x * inv_D  # (BLOCK_HW,)

                    # weight: shape (IC, OC, KD, KH, KW)
                    w_off = ((ic * OC + offs_oc) * KD + kd) * KH * KW + kh * KW + kw
                    wv = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # (BLOCK_OC,)

                    acc += wv[:, None] * mean_x[None, :]

    # add conv bias (per-OC)
    cb = tl.load(cb_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += cb[:, None]

    # store to (B, OC, H, W) layout
    out_off = ((pid_b * OC + offs_oc[:, None]) * H * W) + offs_hw[None, :]
    mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


@triton.jit
def fused_post_kernel(
    x_ptr,        # (B, C, HW)
    bias_ptr,     # (C,)
    out_ptr,      # (B, C, HW)
    B, C, HW,
    scaling_factor,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW

    x_off = pid_b * C * HW + offs_c[:, None] * HW + offs_hw[None, :]
    mask = mask_c[:, None] & mask_hw[None, :]
    x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    v = x + b[:, None]

    v_safe = tl.where(mask_c[:, None], v, -float('inf'))
    m = tl.max(v_safe, axis=0)
    e = tl.exp(v - m[None, :])
    e = tl.where(mask_c[:, None], e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s[None, :]

    t = 2.0 * tl.sigmoid(2.0 * sm) - 1.0
    out = t * scaling_factor

    tl.store(out_ptr + x_off, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        B, IC, D, H, W = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        PAD = self.padding

        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        cbias = self.conv_transpose.bias.contiguous()     # (OC,)

        # Output of conv_transpose+mean is (B, OC, H, W) (stride=1, k=3, p=1 → same spatial)
        pooled = torch.empty((B, OC, H, W), device=x.device, dtype=x.dtype)

        BLOCK_HW = 64
        BLOCK_OC = 32
        grid = (B, triton.cdiv(OC, BLOCK_OC), triton.cdiv(H * W, BLOCK_HW))
        conv_transpose3d_meanpool_kernel[grid](
            x, weight, cbias, pooled,
            B, IC, D, H, W,
            OC, KD, KH, KW,
            PAD, PAD, PAD,
            BLOCK_HW=BLOCK_HW,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # Fused bias + softmax(C) + tanh + scale
        HW = H * W
        out = torch.empty_like(pooled)
        bias_flat = self.bias.view(-1).contiguous()
        BLOCK_C = triton.next_power_of_2(OC)
        BLOCK_HW2 = 8
        grid2 = (B, triton.cdiv(HW, BLOCK_HW2))
        fused_post_kernel[grid2](
            pooled, bias_flat, out,
            B, OC, HW,
            float(self.scaling_factor),
            BLOCK_C=BLOCK_C,
            BLOCK_HW=BLOCK_HW2,
            num_warps=8,
            num_stages=2,
        )
        return out.view(B, OC, 1, H, W)