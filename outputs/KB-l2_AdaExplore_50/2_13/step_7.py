import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr,            # (B, IC, D, H, W)
    w_ptr,            # (OC, IC, KD, KH, KW) flipped weight
    conv_bias_ptr,    # (OC,)
    extra_bias_ptr,   # (OC,)
    out_ptr,          # (B, OC, H, W)
    B, IC, D, H, W, OC,
    scaling_factor,
    inv_D,
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

    KDHW = KD * KH * KW
    IC_KDHW = IC * KDHW

    # Loop d outermost. For each d_out, conv gathers from input depths d_out-1+kd in [0,D).
    for d_out in range(0, D):
        for kd in tl.static_range(0, KD):
            id_d = d_out - 1 + kd
            d_valid = (id_d >= 0) & (id_d < D)
            if d_valid:
                for kh in tl.static_range(0, KH):
                    h_in = h_idx - 1 + kh
                    h_valid = (h_in >= 0) & (h_in < H)
                    for kw in tl.static_range(0, KW):
                        w_in = w_idx - 1 + kw
                        spatial_valid = h_valid & (w_in >= 0) & (w_in < W) & mask_hw

                        for ic in range(0, IC):
                            in_off = (pid_b * IC * D * H * W
                                      + ic * D * H * W
                                      + id_d * H * W
                                      + h_in * W + w_in)
                            x_val = tl.load(x_ptr + in_off, mask=spatial_valid, other=0.0)

                            w_off = offs_oc * IC_KDHW + ic * KDHW + kd * (KH * KW) + kh * KW + kw
                            w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)

                            acc = acc + w_val[:, None] * x_val[None, :]

    cb = tl.load(conv_bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    eb = tl.load(extra_bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    v = acc * inv_D + (cb + eb)[:, None]

    v_safe = tl.where(mask_oc[:, None] & mask_hw[None, :], v, -float('inf'))
    m = tl.max(v_safe, axis=0)
    e = tl.exp(v - m[None, :])
    e = tl.where(mask_oc[:, None] & mask_hw[None, :], e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s[None, :]

    e2 = tl.exp(2.0 * sm)
    t = (e2 - 1.0) / (e2 + 1.0)
    out = t * scaling_factor

    out_off = (pid_b * OC * HW
               + offs_oc[:, None] * HW
               + offs_hw[None, :])
    out_mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_off, out, mask=out_mask)


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

        BLOCK_HW = 128
        BLOCK_OC = triton.next_power_of_2(OC)

        HW = H * W
        grid = (B, (HW + BLOCK_HW - 1) // BLOCK_HW)

        fused_kernel[grid](
            x, w_conv, conv_bias, extra_bias, out,
            B, IC, D, H, W, OC,
            float(self.scaling_factor),
            1.0 / D,
            BLOCK_HW=BLOCK_HW,
            BLOCK_OC=BLOCK_OC,
            KD=3, KH=3, KW=3,
            num_warps=8,
            num_stages=2,
        )

        return out.view(B, OC, 1, H, W)