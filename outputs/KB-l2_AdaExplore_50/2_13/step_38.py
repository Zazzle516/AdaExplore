import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_mean_kernel(
    x_ptr,           # (B, IC, D, H, W)
    w_ptr,           # (IC, OC, KD, KH, KW) - ConvTranspose3d weight layout
    cb_ptr,          # (OC,) - conv bias
    out_ptr,         # (B, OC, H, W)
    B, IC, D, H, W,
    OC,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # program ids: (b, oc_block, hw_block)
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    HW = H * W
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW

    h = offs_hw // W
    w = offs_hw % W

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # accumulator (BLOCK_HW, BLOCK_OC)
    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    # ConvTranspose3d with stride=1: output[od] = sum_{kd} input[od - kd + PAD_D] * w[ic, oc, kd, ...]
    # Equivalently we sum over all valid (id, kd) with id = od - kd + PAD_D, then mean over od.
    # mean over od: (1/D) * sum_{od=0..D-1} sum_{kd} x[id] * w[kd]  where id = od + PAD_D - kd
    # For each (kd, id) pair, count of od values = number of od in [0, D) with od = id + kd - PAD_D
    # That's just 1 if 0 <= id + kd - PAD_D < D. So:
    # sum_{od} sum_{kd} x[id] w[kd] = sum_{id, kd} x[id] w[kd] * 1[0 <= id+kd-PAD_D < D]
    # We loop over (id, kd) pairs.

    # input strides
    # x: (B, IC, D, H, W)
    x_b_stride = IC * D * H * W
    x_ic_stride = D * H * W
    x_d_stride = H * W

    # weight: (IC, OC, KD, KH, KW)
    w_ic_stride = OC * KD * KH * KW
    w_oc_stride = KD * KH * KW
    w_kd_stride = KH * KW
    w_kh_stride = KW

    # For each (kh, kw), compute input h_in, w_in
    for kh in tl.static_range(0, KH):
        h_in = h + PAD_H - kh  # (BLOCK_HW,)
        mask_h = (h_in >= 0) & (h_in < H)
        for kw in tl.static_range(0, KW):
            w_in = w + PAD_W - kw
            mask_w = (w_in >= 0) & (w_in < W)
            mask_hw_in = mask_h & mask_w & mask_hw

            # safe indices
            h_safe = tl.where(mask_h, h_in, 0)
            w_safe = tl.where(mask_w, w_in, 0)
            hw_in_offset = h_safe * W + w_safe  # (BLOCK_HW,)

            # Loop over kd; for each kd, sum over valid id
            for kd in tl.static_range(0, KD):
                # id range: id in [0, D), and id + kd - PAD_D in [0, D)
                # i.e., id in [max(0, PAD_D - kd), min(D, D + PAD_D - kd))
                id_lo = tl.maximum(0, PAD_D - kd)
                id_hi = tl.minimum(D, D + PAD_D - kd)

                # Loop over IC
                for ic in range(0, IC):
                    # weight value: w[ic, offs_oc, kd, kh, kw]
                    w_offs = (ic * w_ic_stride
                              + offs_oc * w_oc_stride
                              + kd * w_kd_stride
                              + kh * w_kh_stride
                              + kw)
                    wv = tl.load(w_ptr + w_offs, mask=mask_oc, other=0.0)  # (BLOCK_OC,)

                    # sum x over id range
                    # accumulate per-element: x[b, ic, id, h_in, w_in]
                    x_base = (pid_b * x_b_stride
                              + ic * x_ic_stride
                              + hw_in_offset)  # (BLOCK_HW,)

                    x_sum = tl.zeros((BLOCK_HW,), dtype=tl.float32)
                    for id_val in range(0, D):
                        valid_id = (id_val >= id_lo) & (id_val < id_hi)
                        x_offs = x_base + id_val * x_d_stride
                        xv = tl.load(x_ptr + x_offs, mask=mask_hw_in & valid_id, other=0.0)
                        x_sum = x_sum + xv

                    # outer product accumulate
                    acc += x_sum[:, None] * wv[None, :]

    # divide by D for mean
    acc = acc / D.to(tl.float32)

    # add conv bias
    cb = tl.load(cb_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + cb[None, :]

    # store: out (B, OC, H, W)
    out_base = pid_b * OC * HW + offs_oc[None, :] * HW + offs_hw[:, None]
    tl.store(out_ptr + out_base, acc, mask=mask_hw[:, None] & mask_oc[None, :])


@triton.jit
def fused_post_kernel(
    x_ptr,         # (B, C, H, W) - mean-pooled conv output (with conv bias already added)
    bias_ptr,      # (C,) - extra bias param
    out_ptr,       # (B, C, H, W)
    B, C, H, W,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    HW = H * W
    b = pid // HW
    rem = pid % HW
    h = rem // W
    w = rem % W

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = b * C * HW + h * W + w
    ptrs = x_ptr + base + offs_c * HW

    x = tl.load(ptrs, mask=mask_c, other=-float('inf'))
    bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    x = x + bias

    x_max = tl.max(x, axis=0)
    x_shift = x - x_max
    e = tl.exp(x_shift)
    e = tl.where(mask_c, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    two_x = 2.0 * sm
    e2 = tl.exp(two_x)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = th * scaling_factor

    out_ptrs = out_ptr + base + offs_c * HW
    tl.store(out_ptrs, out, mask=mask_c)


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
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        # Use cuDNN ConvTranspose3d - fast path. Then fuse mean + bias + softmax + tanh + scale.
        x = self.conv_transpose(x)            # (B, C, D, H, W)
        x = x.mean(dim=2, keepdim=False)      # (B, C, H, W)
        x = x.contiguous()

        B, C, H, W = x.shape
        out = torch.empty_like(x)

        BLOCK_C = 1
        while BLOCK_C < C:
            BLOCK_C *= 2

        bias_flat = self.bias.view(-1).contiguous()

        grid = (B * H * W,)
        fused_post_kernel[grid](
            x, bias_flat, out,
            B, C, H, W,
            float(self.scaling_factor),
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out.unsqueeze(2)