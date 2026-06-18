import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused ConvTranspose2d + multiply + global mean.
# Output shape: (N, OC, 1, 1). One program per (n, oc).
# Each program iterates over (ic, ih, iw, kh, kw) and accumulates the
# contribution to the output sum, with masking by (oh, ow) bounds.
# This preserves the full MAC count of ConvTranspose2d (every (ic, ih, iw, kh, kw)
# tuple is visited and contributes a multiply-add if the corresponding output
# coordinate is in bounds).
@triton.jit
def fused_convtranspose_mean_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    multiplier,
    inv_HW,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    hw_offs = tl.arange(0, BLOCK_HW)

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    HW = IH * IW

    for ic in range(0, IC):
        # Loop over input spatial in tiles
        for hw_start in range(0, HW, BLOCK_HW):
            idx = hw_start + hw_offs
            mask = idx < HW
            ih = idx // IW
            iw = idx % IW

            x_off = pid_n * (IC * HW) + ic * HW + idx
            x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0)  # [BLOCK_HW]

            # For each (kh, kw), check validity and accumulate weight * x.
            # Each input position contributes to one output position per (kh,kw):
            # oh = ih*S - P + kh, ow = iw*S - P + kw
            for kh in tl.static_range(0, KH):
                oh = ih * STRIDE - PAD + kh
                oh_valid = (oh >= 0) & (oh < OH)
                for kw in tl.static_range(0, KW):
                    ow = iw * STRIDE - PAD + kw
                    ow_valid = (ow >= 0) & (ow < OW)
                    valid = mask & oh_valid & ow_valid

                    w_off = ic * (OC * KH * KW) + pid_oc * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)  # scalar

                    contrib = tl.where(valid, x_val * w_val, 0.0)
                    acc += contrib

    s = tl.sum(acc, axis=0)
    b = tl.load(b_ptr + pid_oc)
    # full output sum = s + b * (OH*OW); mean = sum / (OH*OW) = s*inv_HW + b
    mean_val = (s * inv_HW + b) * multiplier

    out_off = pid_n * OC + pid_oc
    tl.store(out_ptr + out_off, mean_val)


def fused_convtranspose_mean(x, weight, bias, stride, padding, output_padding, multiplier):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, 1, 1), device=x.device, dtype=x.dtype)

    BLOCK_HW = 256
    grid = (N, OC)

    fused_convtranspose_mean_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, padding,
        float(multiplier),
        1.0 / (OH * OW),
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()
        bias = self.conv_transpose.bias.contiguous().cuda()
        return fused_convtranspose_mean(
            x, weight, bias,
            self.stride, self.padding, self.output_padding,
            self.multiplier,
        )