import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 32}, num_warps=4, num_stages=2),
    ],
    key=['B', 'IC', 'D', 'H', 'W', 'OC'],
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

    # For each (ic, kh, kw), accumulate weighted sum of x over d positions and kd.
    # Reorder: per (ic, kh, kw), loop over d_in (input depth) and accumulate
    #   x[b,ic,d_in,in_h,in_w] * (Σ_{kd: d_out=d_in-pad+kd in [0,D)} w[ic,oc,kd,kh,kw])
    # This loads x once per (b, ic, d_in, in_h, in_w) and reuses it across all OC and kd.

    for ic in range(IC):
        for kh in range(KH):
            for kw in range(KW):
                in_h = h_idx + PAD_H - kh
                in_w = w_idx + PAD_W - kw
                valid_hw = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W) & mask_hw

                # Preload weights for all kd: shape (BLOCK_OC, KD)
                # w_off[oc, kd] = ic*(OC*KD*KH*KW) + oc*(KD*KH*KW) + kd*(KH*KW) + kh*KW + kw
                # Per-kd:
                # We'll loop over d_in and for each compute Σ_{kd valid} w[..,kd,..]
                for d_in in range(0, D):
                    x_off = (pid_b * IC * D * H * W
                             + ic * D * H * W
                             + d_in * H * W
                             + in_h * W
                             + in_w)
                    x_v = tl.load(x_ptr + x_off, mask=valid_hw, other=0.0)

                    # Build effective weight per OC:
                    # Σ_{kd in [0,KD)} [d_in - PAD_D + kd in [0, D)] * w[ic,oc,kd,kh,kw]
                    # Equivalently kd in [max(0, PAD_D - d_in), min(KD-1, D-1+PAD_D - d_in)]
                    kd_lo = PAD_D - d_in
                    kd_hi = D - 1 + PAD_D - d_in
                    kd_start = tl.maximum(kd_lo, 0)
                    kd_end = tl.minimum(kd_hi, KD - 1)

                    w_eff = tl.zeros((BLOCK_OC,), dtype=tl.float32)
                    for kd in range(0, KD):
                        kd_valid = (kd >= kd_start) & (kd <= kd_end)
                        w_off = ic * (OC * KD * KH * KW) + offs_oc * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=mask_oc & kd_valid, other=0.0,
                                        eviction_policy='evict_last')
                        w_eff = w_eff + w_val

                    acc += w_eff[:, None] * x_v[None, :]

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