import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Custom conv: ConvTranspose3d with stride=1, padding=1, kernel=3 is equivalent
# to Conv3d with flipped weight and same padding=1. We fuse:
#   - conv
#   - mean over D
#   - bias add
#   - softmax over C
#   - tanh
#   - scaling
#
# Strategy: one program per (n, h_tile, w_tile). Each program computes the
# full channel column for its spatial tile by accumulating the conv output
# summed (then divided) over D in registers. Channels become the M-dim of
# the reduction, so softmax can be done across channels in registers.


@triton.jit
def fused_convT3d_mean_softmax_tanh_kernel(
    x_ptr,            # (B, IC, D, H, W) input
    w_ptr,            # flipped conv weight (OC, IC, KD, KH, KW)
    conv_bias_ptr,    # (OC,) original conv_transpose bias
    extra_bias_ptr,   # (OC,) extra bias parameter
    out_ptr,          # (B, OC, H, W)
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

    # acc shape: (BLOCK_OC, BLOCK_HW) -- sum over D of conv output
    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    inv_D = 1.0 / D

    # Loop over input channels and kernel dims
    for ic in range(0, IC):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    # input coords: h_in = h - 1 + kh, w_in = w - 1 + kw
                    # For each output h,w we sum over D positions d.
                    # Conv (with padding 1): out[d] = sum over kd of in[d-1+kd] * w[kd]
                    # Summing over d (with valid d in [0,D)):
                    #   sum_d in[d-1+kd] = sum over valid in_d in [0, D)
                    # which equals: sum of in[id] for id in [max(0,kd-1), min(D, D+kd-1))
                    # Equivalently: count_d_for_id = number of d in [0,D) s.t. d-1+kd == id
                    # For each id, it's 1 if 0 <= id < D and 0 <= id - kd + 1 < D
                    # i.e., id in [max(0,kd-1), min(D-1, D-2+kd)] inclusive.
                    # Just compute via loop over d:
                    h_in = h_idx - 1 + kh
                    w_in = w_idx - 1 + kw
                    spatial_valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask_hw

                    # Load weight slice (OC, IC=fixed, kd, kh, kw)
                    w_off = offs_oc * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # (BLOCK_OC,)

                    # Sum input over depth-axis with appropriate offset
                    # For a fixed kd, the d-axis sum reduces to summing input
                    # over input depth indices id where 0 <= id - kd + 1 < D and 0 <= id < D
                    # Compute the depth-sum of input at (n, ic, :, h_in, w_in)
                    # restricted to valid range based on kd.
                    id_start = tl.maximum(0, kd - 1)
                    id_end = tl.minimum(D, D + kd - 1)  # exclusive

                    in_sum = tl.zeros((BLOCK_HW,), dtype=tl.float32)
                    for d_iter in range(0, D):
                        d_valid = (d_iter >= id_start) & (d_iter < id_end)
                        in_off = (pid_b * IC * D * H * W
                                  + ic * D * H * W
                                  + d_iter * H * W
                                  + h_in * W + w_in)
                        in_mask = spatial_valid & d_valid
                        x_val = tl.load(x_ptr + in_off, mask=in_mask, other=0.0)
                        in_sum = in_sum + x_val

                    # outer product: w_val (BLOCK_OC,) * in_sum (BLOCK_HW,) -> (BLOCK_OC, BLOCK_HW)
                    acc = acc + w_val[:, None] * in_sum[None, :]

    # Now acc = sum_d conv_out[n, oc, d, h, w] (without bias)
    # mean = acc / D + conv_bias (mean adds conv_bias since it's constant over d)
    # Then add extra_bias
    cb = tl.load(conv_bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    eb = tl.load(extra_bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    v = acc * inv_D + (cb + eb)[:, None]  # (BLOCK_OC, BLOCK_HW)

    # Softmax over channels (axis=0)
    v_safe = tl.where(mask_oc[:, None] & mask_hw[None, :], v, -float('inf'))
    m = tl.max(v_safe, axis=0)  # (BLOCK_HW,)
    e = tl.exp(v - m[None, :])
    e = tl.where(mask_oc[:, None] & mask_hw[None, :], e, 0.0)
    s = tl.sum(e, axis=0)  # (BLOCK_HW,)
    sm = e / s[None, :]

    # tanh
    t = (tl.exp(sm) - tl.exp(-sm)) / (tl.exp(sm) + tl.exp(-sm))
    out = t * scaling_factor

    # Store: out shape (B, OC, H, W)
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

        # Precompute flipped weight at init time (treat as constant relative to inputs).
        # ConvTranspose3d weight shape: (in_channels, out_channels, kD, kH, kW)
        # Equivalent Conv3d weight shape: (out_channels, in_channels, kD, kH, kW)
        # with flipped spatial dims.
        with torch.no_grad():
            w = self.conv_transpose.weight.data  # (IC, OC, kD, kH, kW)
            w_flipped = torch.flip(w, dims=[2, 3, 4])
            w_conv = w_flipped.permute(1, 0, 2, 3, 4).contiguous()  # (OC, IC, kD, kH, kW)
            self.register_buffer('w_conv_buf', w_conv.clone())

    def _get_conv_weight(self):
        # Recompute from current parameter (in case of training); for eval this
        # is constant but kept correct for safety.
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

        # Use precomputed flipped weight (it's the same param values; safe at eval).
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

        fused_convT3d_mean_softmax_tanh_kernel[grid](
            x, w_conv, conv_bias, extra_bias, out,
            B, IC, D, H, W, OC,
            float(self.scaling_factor),
            BLOCK_HW=BLOCK_HW,
            BLOCK_OC=BLOCK_OC,
            KD=3, KH=3, KW=3,
            num_warps=4,
            num_stages=2,
        )

        return out.view(B, OC, 1, H, W)