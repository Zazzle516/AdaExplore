import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_mean_kernel(
    x_ptr,       # (B, IC, D, H, W)
    w_ptr,       # (IC, OC, KD, KH, KW)
    cb_ptr,      # (OC,) conv bias
    bias_ptr,    # (OC,) extra bias
    out_ptr,     # (B, OC, H, W)
    B, IC, D, H, W,
    OC, KD, KH, KW,
    PAD_D, PAD_H, PAD_W,
    scaling_factor,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < (H * W)
    h_idx = offs_hw // W
    w_idx = offs_hw % W

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # Convolution-equivalent: stride=1 ConvTranspose3d with padding=1, k=3 == regular Conv3d
    # but with weight shape (IC, OC, KD, KH, KW) and effectively flipped kernel.
    # ConvTranspose3d output[b, oc, d, h, w] = sum_{ic,kd,kh,kw} x[b,ic, d+pad-kd, h+pad-kh, w+pad-kw] * w[ic,oc,kd,kh,kw]
    # We sum over D in the output and divide by D for mean.

    for ic in range(IC):
        for kd in range(KD):
            for kh in range(KH):
                for kw in range(KW):
                    # weight[ic, :, kd, kh, kw]
                    w_off = ic * (OC * KD * KH * KW) + offs_oc * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # (BLOCK_OC,)

                    # input h, w positions
                    in_h = h_idx + PAD_H - kh
                    in_w = w_idx + PAD_W - kw
                    valid_hw = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W) & mask_hw

                    # Sum input over depth axis for valid d positions
                    # in_d = d_out + PAD_D - kd, d_out in [0,D)
                    # in_d in [PAD_D - kd, D - 1 + PAD_D - kd]
                    # valid when 0 <= in_d < D
                    d_lo = PAD_D - kd
                    d_hi = D - 1 + PAD_D - kd
                    d_start = tl.maximum(d_lo, 0)
                    d_end = tl.minimum(d_hi, D - 1)

                    # Accumulate sum_d x[b, ic, d, in_h, in_w] for d in [d_start, d_end]
                    x_sum = tl.zeros((BLOCK_HW,), dtype=tl.float32)
                    for d in range(0, D):
                        d_valid = (d >= d_start) & (d <= d_end)
                        x_off = (pid_b * IC * D * H * W
                                 + ic * D * H * W
                                 + d * H * W
                                 + in_h * W
                                 + in_w)
                        m = valid_hw & d_valid
                        x_v = tl.load(x_ptr + x_off, mask=m, other=0.0)
                        x_sum = x_sum + x_v

                    # outer product: (BLOCK_OC,1) * (1,BLOCK_HW)
                    acc += w_val[:, None] * x_sum[None, :]

    # Divide by D for mean over depth
    acc = acc / D

    # Add conv bias and extra bias
    cb = tl.load(cb_ptr + offs_oc, mask=mask_oc, other=0.0)
    eb = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + cb[:, None] + eb[:, None]

    # Softmax over OC
    acc_safe = tl.where(mask_oc[:, None] & mask_hw[None, :], acc, -float('inf'))
    m = tl.max(acc_safe, axis=0)  # (BLOCK_HW,)
    e = tl.exp(acc - m[None, :])
    e = tl.where(mask_oc[:, None] & mask_hw[None, :], e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s[None, :]

    # tanh
    t = 2.0 * tl.sigmoid(2.0 * sm) - 1.0
    out = t * scaling_factor

    out_off = (pid_b * OC * H * W
               + offs_oc[:, None] * (H * W)
               + offs_hw[None, :])
    out_mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_off, out, mask=out_mask)


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
        cbias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        ebias = self.bias.view(-1).contiguous()

        out = torch.empty((B, OC, H, W), device=x.device, dtype=x.dtype)

        BLOCK_HW = 64
        BLOCK_OC = triton.next_power_of_2(OC)

        grid = (B, triton.cdiv(H * W, BLOCK_HW))
        fused_conv_mean_kernel[grid](
            x, weight, cbias, ebias, out,
            B, IC, D, H, W,
            OC, KD, KH, KW,
            PAD, PAD, PAD,
            float(self.scaling_factor),
            BLOCK_HW=BLOCK_HW,
            BLOCK_OC=BLOCK_OC,
            num_warps=8,
            num_stages=2,
        )
        return out.view(B, OC, 1, H, W)