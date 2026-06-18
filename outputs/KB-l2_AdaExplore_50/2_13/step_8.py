import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_mean_bias_kernel(
    x_ptr, w_ptr, conv_bias_ptr, extra_bias_ptr, out_ptr,
    B, IC, D, H, W,
    OC, KD, KH, KW,
    PAD_D, PAD_H, PAD_W,
    inv_D,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (B, ceil(OC/BLOCK_OC), ceil(H*W/BLOCK_HW))
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    HW = H * W

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    mask_oc = offs_oc < OC
    mask_hw = offs_hw < HW

    h = offs_hw // W
    w = offs_hw % W

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # Loop over kd, kh, kw, ic, d
    # weight layout (Conv3d): (OC, IC, KD, KH, KW)
    for kd in range(KD):
        for kh in range(KH):
            for kw in range(KW):
                ih = h + kh - PAD_H
                iw = w + kw - PAD_W
                mask_h = (ih >= 0) & (ih < H)
                mask_w = (iw >= 0) & (iw < W)
                spatial_mask = mask_h & mask_w & mask_hw

                # accumulate sum over d of x[b, ic, d+kd-pad_d, ih, iw]
                # Per (kh,kw,kd), we still loop over IC and over D (output d)
                for ic in range(IC):
                    # weight value: w[oc, ic, kd, kh, kw]
                    w_offs = offs_oc * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    wv = tl.load(w_ptr + w_offs, mask=mask_oc, other=0.0)  # (BLOCK_OC,)

                    # sum_d x[b,ic, d+kd-pad_d, ih, iw] over d=0..D-1
                    sum_x = tl.zeros((BLOCK_HW,), dtype=tl.float32)
                    for d in range(D):
                        idd = d + kd - PAD_D
                        m_d = (idd >= 0) & (idd < D)
                        x_offs = pid_b * (IC * D * H * W) + ic * (D * H * W) + idd * (H * W) + ih * W + iw
                        x_mask = spatial_mask & m_d
                        xv = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)
                        sum_x = sum_x + xv

                    acc = acc + wv[:, None] * sum_x[None, :]

    # mean over D
    acc = acc * inv_D

    # add conv bias
    cb = tl.load(conv_bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    eb = tl.load(extra_bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + cb[:, None] + eb[:, None]

    # store to (B, OC, H, W) layout - we'll do softmax separately
    out_offs = pid_b * (OC * HW) + offs_oc[:, None] * HW + offs_hw[None, :]
    out_mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


@triton.jit
def softmax_tanh_scale_kernel(
    x_ptr, out_ptr,
    B, C, HW,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // HW
    hw = pid % HW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = b * C * HW + hw
    ptrs = x_ptr + base + offs_c * HW

    x = tl.load(ptrs, mask=mask_c, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask_c, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    e2 = tl.exp(2.0 * sm)
    t = (e2 - 1.0) / (e2 + 1.0)
    out = t * scaling_factor

    tl.store(out_ptr + base + offs_c * HW, out, mask=mask_c)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        # Build a ConvTranspose3d to mirror reference parameter init/shapes.
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)

    def forward(self, x):
        # ConvTranspose3d weight has shape (IC, OC, KD, KH, KW).
        # With stride=1, padding=p, ConvTranspose3d is equivalent to Conv3d with
        # weight transposed to (OC, IC, KD, KH, KW) and flipped along spatial dims,
        # and padding = K - 1 - p.
        B, IC, D, H, W = x.shape
        OC = self.out_channels
        KD, KH, KW = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding

        assert sd == 1 and sh == 1 and sw == 1, "Only stride=1 supported in this fused kernel"

        # Build equivalent conv weight: (OC, IC, KD, KH, KW), flipped
        wt = self.conv_transpose.weight  # (IC, OC, KD, KH, KW)
        wt = wt.permute(1, 0, 2, 3, 4).contiguous()
        wt = torch.flip(wt, dims=[2, 3, 4]).contiguous()

        conv_bias = self.conv_transpose.bias  # (OC,)
        extra_bias = self.bias.view(-1)  # (OC,)

        # Equivalent padding for conv form
        eff_pd = KD - 1 - pd
        eff_ph = KH - 1 - ph
        eff_pw = KW - 1 - pw

        x = x.contiguous()
        # Output (B, OC, H, W) — depth has been reduced
        out_pre = torch.empty((B, OC, H, W), device=x.device, dtype=x.dtype)

        BLOCK_HW = 128
        BLOCK_OC = 32
        HW = H * W
        grid = (B, triton.cdiv(OC, BLOCK_OC), triton.cdiv(HW, BLOCK_HW))

        conv3d_mean_bias_kernel[grid](
            x, wt, conv_bias, extra_bias, out_pre,
            B, IC, D, H, W,
            OC, KD, KH, KW,
            eff_pd, eff_ph, eff_pw,
            1.0 / D,
            BLOCK_HW=BLOCK_HW,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # softmax over channels + tanh + scale
        # Reshape to (B, C, 1, H, W) layout expected by softmax kernel
        out = torch.empty_like(out_pre)
        grid2 = (B * HW,)
        BLOCK_C = triton.next_power_of_2(OC)
        softmax_tanh_scale_kernel[grid2](
            out_pre, out,
            B, OC, HW,
            self.scaling_factor,
            BLOCK_C=BLOCK_C,
            num_warps=2,
        )
        return out.view(B, OC, 1, H, W)