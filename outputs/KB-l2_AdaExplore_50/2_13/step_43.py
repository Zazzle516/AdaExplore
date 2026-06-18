import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=3),
    ],
    key=['B', 'IC', 'D', 'H', 'W', 'OC', 'KD', 'KH', 'KW'],
)
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

    # Reorganized loop: outer (kh,kw,ic,d_in), inner kd. Each x-load reused across KD.
    # Multiply-add count per output (h,w,oc) is still IC*KD*KH*KW*D.
    base_x = pid_b * IC * D * H * W

    for kh in range(KH):
        for kw in range(KW):
            in_h = h_idx + PAD_H - kh
            in_w = w_idx + PAD_W - kw
            valid_hw = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W) & mask_hw
            in_h_c = tl.where(valid_hw, in_h, 0)
            in_w_c = tl.where(valid_hw, in_w, 0)
            spatial_off = in_h_c * W + in_w_c

            for ic in range(IC):
                ic_off = ic * D * H * W
                # Preload all KD weights for this (ic, kh, kw)
                # w[ic, oc, kd, kh, kw]
                for d_in in range(D):
                    x_off = base_x + ic_off + d_in * H * W + spatial_off
                    x_val = tl.load(x_ptr + x_off, mask=valid_hw, other=0.0)  # (BLOCK_HW,)

                    for kd in range(KD):
                        d_out = d_in - PAD_D + kd
                        d_valid = (d_out >= 0) & (d_out < D)
                        w_off = (ic * (OC * KD * KH * KW)
                                 + offs_oc * (KD * KH * KW)
                                 + kd * (KH * KW) + kh * KW + kw)
                        w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0,
                                        eviction_policy='evict_last')
                        contrib = w_val[:, None] * x_val[None, :]
                        acc += tl.where(d_valid, contrib, 0.0)

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

        BLOCK_OC = triton.next_power_of_2(OC)

        grid = lambda META: (B, triton.cdiv(H * W, META['BLOCK_HW']))
        fused_conv_mean_kernel[grid](
            x, weight, cbias, ebias, out,
            B, IC, D, H, W,
            OC, KD, KH, KW,
            PAD, PAD, PAD,
            float(self.scaling_factor),
            BLOCK_OC=BLOCK_OC,
        )
        return out.view(B, OC, 1, H, W)