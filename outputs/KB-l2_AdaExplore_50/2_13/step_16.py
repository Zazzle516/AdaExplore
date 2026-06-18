import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_convT3d_mean_softmax_tanh_kernel(
    x_ptr,            # (B, IC, D, H, W) input
    w_ptr,            # flipped conv weight (OC, IC, KD, KH, KW)
    conv_bias_ptr,    # (OC,)
    extra_bias_ptr,   # (OC,)
    out_ptr,          # (B, OC, H, W)
    x_sum_ptr,        # (B, IC, H, W) precomputed sum over D ; not used here, but kept for compat
    B, IC, D, H, W,
    OC,
    scaling_factor,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    HW = H * W
    hw_start = pid_hw * BLOCK_HW
    offs_hw = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW

    h_idx = offs_hw // W
    w_idx = offs_hw % W

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    inv_D = 1.0 / D

    # Loop over input channels and kernel spatial dims (KH, KW).
    # For each (kh, kw), we need: sum_kd w[kd,kh,kw] * (sum over valid id of x[id, h_in, w_in])
    # We'll compute this by iterating kd and using the precomputed depth-sums for each kd's valid range.
    # 
    # The valid id range for kd in {0,1,2} (with D out depth = D, padding=1):
    #   kd=0: id in [0, D-1)  -> x[0..D-2]   (length D-1)  but careful: 
    #         Actually: d-1+kd = id with d in [0,D), kd=0 => id = d-1, valid id in [0, D-1) when d in [1, D)
    #         So id in [0, D-2] inclusive, sum of x[0..D-2]
    #   kd=1: id in [0, D-1] inclusive  -> full sum
    #   kd=2: id in [1, D-1] inclusive  -> sum of x[1..D-1]
    #
    # We'll compute three depth-sums per (ic, h_in, w_in): full, drop_first, drop_last.

    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                h_in = h_idx - 1 + kh
                w_in = w_idx - 1 + kw
                spatial_valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask_hw

                # Compute depth sums S_full, S_minus_first, S_minus_last
                base_off = pid_b * IC * D * H * W + ic * D * H * W + h_in * W + w_in

                S_full = tl.zeros((BLOCK_HW,), dtype=tl.float32)
                for d_iter in range(0, D):
                    in_off = base_off + d_iter * H * W
                    x_val = tl.load(x_ptr + in_off, mask=spatial_valid, other=0.0)
                    S_full = S_full + x_val

                # x at d=0 and d=D-1
                x_first_off = base_off + 0
                x_last_off = base_off + (D - 1) * H * W
                x_first = tl.load(x_ptr + x_first_off, mask=spatial_valid, other=0.0)
                x_last = tl.load(x_ptr + x_last_off, mask=spatial_valid, other=0.0)

                S_kd0 = S_full - x_last      # sum x[0..D-2]
                S_kd1 = S_full
                S_kd2 = S_full - x_first     # sum x[1..D-1]

                # Load weight slices for kd=0,1,2 at fixed (ic, kh, kw)
                w_base = offs_oc * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kh * KW + kw
                w_kd0 = tl.load(w_ptr + w_base + 0 * (KH * KW), mask=mask_oc, other=0.0)
                w_kd1 = tl.load(w_ptr + w_base + 1 * (KH * KW), mask=mask_oc, other=0.0)
                w_kd2 = tl.load(w_ptr + w_base + 2 * (KH * KW), mask=mask_oc, other=0.0)

                acc += w_kd0[:, None] * S_kd0[None, :]
                acc += w_kd1[:, None] * S_kd1[None, :]
                acc += w_kd2[:, None] * S_kd2[None, :]

    cb = tl.load(conv_bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    eb = tl.load(extra_bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    v = acc * inv_D + (cb + eb)[:, None]

    full_mask = mask_oc[:, None] & mask_hw[None, :]
    v_safe = tl.where(full_mask, v, -float('inf'))
    m = tl.max(v_safe, axis=0)
    e = tl.exp(v - m[None, :])
    e = tl.where(full_mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s[None, :]

    # tanh via sigmoid
    t = 2.0 * tl.sigmoid(2.0 * sm) - 1.0
    out = t * scaling_factor

    out_off = pid_b * OC * HW + offs_oc[:, None] * HW + offs_hw[None, :]
    tl.store(out_ptr + out_off, out, mask=full_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        with torch.no_grad():
            w = self.conv_transpose.weight.data  # (IC, OC, kD, kH, kW)
            w_flipped = torch.flip(w, dims=[2, 3, 4])
            w_conv = w_flipped.permute(1, 0, 2, 3, 4).contiguous()
            self.register_buffer('w_conv_buf', w_conv.clone())

    def _get_conv_weight(self):
        w = self.conv_transpose.weight
        w_flipped = torch.flip(w, dims=[2, 3, 4])
        w_conv = w_flipped.permute(1, 0, 2, 3, 4).contiguous()
        return w_conv

    def forward(self, x):
        assert self.stride == 1 and self.padding == 1 and self.kernel_size == 3, \
            "Custom kernel assumes stride=1, padding=1, kernel=3"

        x = x.contiguous()
        B, IC, D, H, W = x.shape
        OC = self.out_channels

        if self.training:
            w_conv = self._get_conv_weight()
        else:
            w_conv = self.w_conv_buf

        conv_bias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None \
            else torch.zeros(OC, device=x.device, dtype=x.dtype)
        extra_bias = self.bias.view(-1).contiguous()

        out = torch.empty((B, OC, H, W), device=x.device, dtype=x.dtype)

        BLOCK_HW = 64
        BLOCK_OC = triton.next_power_of_2(OC)

        HW = H * W
        grid = (B, (HW + BLOCK_HW - 1) // BLOCK_HW)

        # Dummy x_sum tensor (unused) for parameter compatibility
        x_sum = torch.empty(1, device=x.device, dtype=x.dtype)

        fused_convT3d_mean_softmax_tanh_kernel[grid](
            x, w_conv, conv_bias, extra_bias, out, x_sum,
            B, IC, D, H, W, OC,
            float(self.scaling_factor),
            BLOCK_HW=BLOCK_HW,
            BLOCK_OC=BLOCK_OC,
            KD=3, KH=3, KW=3,
            num_warps=4,
            num_stages=2,
        )

        return out.view(B, OC, 1, H, W)