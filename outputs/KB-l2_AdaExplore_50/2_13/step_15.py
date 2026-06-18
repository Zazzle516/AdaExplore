import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv_mean_kernel(
    x_ptr,        # (B, IC, D, H, W)
    w_ptr,        # (IC, OC, KD, KH, KW)  - ConvTranspose3d weight layout
    cb_ptr,       # (OC,) - conv bias
    bias_ptr,     # (OC,) - extra bias
    out_ptr,      # (B, OC, H, W) - already mean-pooled over D
    scaling_factor,
    B, IC, D, H, W,
    OC,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    OC_PAD: tl.constexpr,
):
    # Each program: one (b, hw_tile) — produces softmax/tanh/scale output for all OC at those positions.
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    HW = H * W
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW
    h = offs_hw // W
    w = offs_hw % W

    offs_oc = tl.arange(0, OC_PAD)
    mask_oc = offs_oc < OC

    # Accumulator: (OC_PAD, BLOCK_HW)
    acc = tl.zeros((OC_PAD, BLOCK_HW), dtype=tl.float32)

    # ConvTranspose3d with stride=1, padding=1, kernel=3 is equivalent to:
    # out[b, oc, d, h, w] = sum_{ic, kd, kh, kw} x[b, ic, d - kd + PAD_D, h - kh + PAD_H, w - kw + PAD_W] * w_flipped[ic, oc, kd, kh, kw]
    # ConvTranspose3d weight is (IC, OC, KD, KH, KW) and uses flipped kernel relative to conv.
    # Precisely: out[b,oc,d,h,w] = sum_{ic,kd,kh,kw} x[b,ic, d+kd-PAD_D, h+kh-PAD_H, w+kw-PAD_W] * weight[ic,oc, KD-1-kd, KH-1-kh, KW-1-kw]
    # We'll loop over kd, kh, kw and ic.

    # Sum over D inside kernel: out_mean[b,oc,h,w] = (1/D) * sum_d out[b,oc,d,h,w]
    # = (1/D) * sum_{kd,kh,kw,ic} weight_flipped[...] * sum_d x[b, ic, d+kd-PAD_D, h+kh-PAD_H, w+kw-PAD_W]
    # But we cannot pre-reduce x over d (safety contract). So we keep d in the loop.

    inv_D = 1.0 / D.to(tl.float32)

    for kd in tl.static_range(0, KD):
        for kh in tl.static_range(0, KH):
            for kw_ in tl.static_range(0, KW):
                # input spatial position
                ih = h + kh - PAD_H
                iw = w + kw_ - PAD_W
                in_h_valid = (ih >= 0) & (ih < H)
                in_w_valid = (iw >= 0) & (iw < W)
                hw_valid = mask_hw & in_h_valid & in_w_valid

                # weight index for flipped kernel
                wkd = KD - 1 - kd
                wkh = KH - 1 - kh
                wkw = KW - 1 - kw_

                # Loop over D
                for d in range(0, D):
                    id_ = d + kd - PAD_D
                    d_valid = (id_ >= 0) & (id_ < D)
                    if d_valid:
                        # Loop over IC
                        for ic in range(0, IC):
                            # Load x[b, ic, id_, ih, iw] for all hw
                            x_off = ((pid_b * IC + ic) * D + id_) * H * W + ih * W + iw
                            x_val = tl.load(x_ptr + x_off, mask=hw_valid, other=0.0)
                            # Load weight[ic, :, wkd, wkh, wkw]
                            w_off = ((ic * OC + offs_oc) * KD + wkd) * KH * KW + wkh * KW + wkw
                            w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)
                            # acc += w_val[:, None] * x_val[None, :]
                            acc += w_val[:, None] * x_val[None, :]

    # Multiply by 1/D for mean
    acc = acc * inv_D

    # Add conv bias
    cb = tl.load(cb_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + cb[:, None]

    # Add extra bias
    eb = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + eb[:, None]

    # Softmax over OC dimension
    acc = tl.where(mask_oc[:, None] & mask_hw[None, :], acc, -float('inf'))
    m = tl.max(acc, axis=0)
    e = tl.exp(acc - m[None, :])
    e = tl.where(mask_oc[:, None] & mask_hw[None, :], e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s[None, :]

    # tanh
    e2 = tl.exp(2.0 * sm)
    t = (e2 - 1.0) / (e2 + 1.0)
    out = t * scaling_factor

    # Store: (B, OC, H, W)
    out_off = (pid_b * OC + offs_oc[:, None]) * HW + offs_hw[None, :]
    tl.store(out_ptr + out_off, out, mask=mask_oc[:, None] & mask_hw[None, :])


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
        # Use cuDNN for conv_transpose
        x = self.conv_transpose(x)  # (B, C, D, H, W)
        x = x.contiguous()

        B, C, D, H, W = x.shape
        # Output shape (B, C, 1, H, W) — directly allocate
        out = torch.empty((B, C, 1, H, W), dtype=x.dtype, device=x.device)

        bias_flat = self.bias.view(-1).contiguous()

        BLOCK_C = 1
        while BLOCK_C < C:
            BLOCK_C *= 2

        BLOCK_HW = 128
        HW = H * W
        grid = (B, (HW + BLOCK_HW - 1) // BLOCK_HW)
        _post_kernel[grid](
            x, bias_flat, out,
            B, C, D, H, W,
            float(self.scaling_factor),
            BLOCK_C=BLOCK_C,
            BLOCK_HW=BLOCK_HW,
            num_warps=8,
            num_stages=2,
        )

        return out


@triton.jit
def _post_kernel(
    x_ptr, bias_ptr, out_ptr,
    B, C, D, H, W,
    scaling_factor,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)
    HW = H * W

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    mask2d = mask_c[:, None] & mask_hw[None, :]

    # Accumulate over D
    acc = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)
    base = pid_b * C * D * HW + offs_c[:, None] * D * HW + offs_hw[None, :]
    for d in range(0, D):
        x_ptrs = x_ptr + base + d * HW
        x = tl.load(x_ptrs, mask=mask2d, other=0.0).to(tl.float32)
        acc += x

    inv_D = 1.0 / D.to(tl.float32)
    acc = acc * inv_D

    bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)

    v = acc + bias[:, None]
    v = tl.where(mask_c[:, None], v, -float('inf'))

    m = tl.max(v, axis=0)
    e = tl.exp(v - m[None, :])
    e = tl.where(mask_c[:, None], e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s[None, :]

    e2 = tl.exp(2.0 * sm)
    t = (e2 - 1.0) / (e2 + 1.0)
    out = t * scaling_factor

    # Output is (B, C, 1, H, W) - same stride pattern as (B, C, H, W)
    out_ptrs = out_ptr + pid_b * C * HW + offs_c[:, None] * HW + offs_hw[None, :]
    tl.store(out_ptrs, out, mask=mask2d)