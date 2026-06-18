import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_mean_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N, OC, ceil(OH*OW / BLOCK_HW))
    n = tl.program_id(0)
    oc = tl.program_id(1)
    hw_blk = tl.program_id(2)

    offs_hw = hw_blk * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < (OH * OW)
    oh = offs_hw // OW
    ow = offs_hw % OW

    # ConvTranspose3d with stride=1, padding=1, kernel=3:
    # y[oc, od, oh, ow] = sum_{ic, kd, kh, kw} x[ic, od + PAD_D - kd, oh + PAD_H - kh, ow + PAD_W - kw] * w[ic, oc, kd, kh, kw]
    # This is equivalent to a regular conv with flipped kernel.
    # Since we mean over od, we sum over od=0..OD-1 then divide by OD.

    bias_val = tl.load(b_ptr + oc).to(tl.float32)
    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    # Iterate over input channels and kernel spatial (kd, kh, kw)
    # for each output (oh, ow) and sum over od.
    # For each kd, the input depth id = od + PAD_D - kd, where od ranges 0..OD-1.
    # Sum over od of x[id] where id valid (0 <= id < D).
    # Since stride=1 padding=PAD_D, OD = D + 2*PAD_D - KD + 1. For PAD_D=1, KD=3: OD = D.
    # id range: od - kd + PAD_D, od in [0, OD). For kd=0: id in [PAD_D, OD+PAD_D-1] -> [1, D]
    # For kd=1: id in [0, D-1]; for kd=2: id in [-1, D-2] -> [0, D-2]
    # So sum over od of x[id] for valid id is: sum over id in valid range.

    # Compute weight pointer base
    # w shape: (IC, OC, KD, KH, KW), stride (OC*KD*KH*KW, KD*KH*KW, KH*KW, KW, 1)
    w_oc_stride = KD * KH * KW
    w_ic_stride = OC * w_oc_stride

    for ic in range(0, IC):
        for kd in range(0, KD):
            # Determine valid id range for this kd
            # id = od - kd + PAD_D, od in [0, OD)
            id_min = -kd + PAD_D  # od=0
            id_max = OD - 1 - kd + PAD_D  # od=OD-1
            # Clamp to [0, D-1]
            lo = id_min if id_min > 0 else 0
            hi = id_max if id_max < D - 1 else D - 1

            for kh in range(0, KH):
                for kw in range(0, KW):
                    ih = oh - kh + PAD_H
                    iw = ow - kw + PAD_W
                    mask_h = (ih >= 0) & (ih < H)
                    mask_w = (iw >= 0) & (iw < W)
                    mask_spatial = mask_h & mask_w & mask_hw

                    w_off = ic * w_ic_stride + oc * w_oc_stride + kd * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off).to(tl.float32)

                    # Sum input over id in [lo, hi]
                    # Pre-compute base pointer for (n, ic, *, ih, iw)
                    # x stride: (IC*D*H*W, D*H*W, H*W, W, 1)
                    x_base = n * IC * D * H * W + ic * D * H * W + ih * W + iw
                    # Loop over id
                    sum_x = tl.zeros((BLOCK_HW,), dtype=tl.float32)
                    for idd in range(lo, hi + 1):
                        x_off = x_base + idd * H * W
                        v = tl.load(x_ptr + x_off, mask=mask_spatial, other=0.0)
                        sum_x += v.to(tl.float32)

                    acc += sum_x * w_val

    # Divide by OD for mean, add bias
    acc = acc / OD + bias_val

    # Store to output: (N, OC, 1, OH, OW)
    out_off = n * OC * OH * OW + oc * OH * OW + offs_hw
    tl.store(out_ptr + out_off, acc, mask=mask_hw)


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


def fused_softmax_tanh_scale(x, scaling_factor):
    B, C, D, H, W = x.shape
    assert D == 1
    HW = H * W
    x = x.contiguous()
    out = torch.empty_like(x)
    BLOCK_C = triton.next_power_of_2(C)
    grid = (B * HW,)
    softmax_tanh_scale_kernel[grid](
        x, out, B, C, HW, scaling_factor,
        BLOCK_C=BLOCK_C, num_warps=2,
    )
    return out


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
        N, IC, D, H, W = x.shape
        OC = self.out_channels
        K = self.kernel_size
        P = self.padding
        # ConvTranspose3d output: OD = (D-1)*stride - 2*pad + K = D - 2P + K - 1 = D for stride=1,pad=1,K=3
        OD = (D - 1) * self.stride - 2 * P + K
        OH = (H - 1) * self.stride - 2 * P + K
        OW = (W - 1) * self.stride - 2 * P + K

        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, K, K, K)
        conv_bias = self.conv_transpose.bias.contiguous()  # (OC,)

        # Output of conv + mean over D: (N, OC, 1, OH, OW)
        mean_out = torch.empty((N, OC, 1, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_HW = 128
        grid = (N, OC, triton.cdiv(OH * OW, BLOCK_HW))
        conv_mean_kernel[grid](
            x, weight, conv_bias, mean_out,
            N, IC, D, H, W,
            OC, OD, OH, OW,
            K, K, K,
            P, P, P,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=2,
        )

        # Add per-channel bias and apply softmax+tanh+scale
        mean_out = mean_out + self.bias
        out = fused_softmax_tanh_scale(mean_out, self.scaling_factor)
        return out