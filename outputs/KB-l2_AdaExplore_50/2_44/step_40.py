import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Scatter-style conv-transpose fused with multiply and sum-reduction over (H, W).
# Each program handles a tile of (N, IC, IH*IW). It accumulates partial sums
# per (n, oc) into a small (N, OC) buffer via atomic adds.
# The full output is materialized into out_full so that asymptotic MAC count
# matches the reference (we both store full output AND accumulate sum).
@triton.jit
def conv_transpose_scatter_kernel(
    x_ptr, w_ptr, out_full_ptr,
    N, IC, IH, IW, OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_ic = tl.program_id(1)
    pid_hw = tl.program_id(2)

    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < (IH * IW)
    ih = hw_offs // IW
    iw = hw_offs % IW

    # Load x[n, ic, :] tile
    x_offs = pid_n * (IC * IH * IW) + pid_ic * (IH * IW) + hw_offs
    x_val = tl.load(x_ptr + x_offs, mask=hw_mask, other=0.0)  # [BLOCK_HW]

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    for kh in tl.static_range(0, KH):
        oh = ih * STRIDE - PAD + kh  # [BLOCK_HW]
        oh_valid = (oh >= 0) & (oh < OH)
        for kw in tl.static_range(0, KW):
            ow = iw * STRIDE - PAD + kw
            ow_valid = (ow >= 0) & (ow < OW)
            valid = hw_mask & oh_valid & ow_valid  # [BLOCK_HW]

            # weight[ic, :, kh, kw] -> [BLOCK_OC]
            w_off = pid_ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
            w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

            # contrib = w[oc] * x[hw] -> [BLOCK_OC, BLOCK_HW]
            contrib = w_val[:, None] * x_val[None, :]

            # scatter-add into out_full[n, oc, oh, ow]
            out_off = (pid_n * OC * OH * OW
                       + oc_offs[:, None] * (OH * OW)
                       + oh[None, :] * OW
                       + ow[None, :])
            mask2d = oc_mask[:, None] & valid[None, :]
            tl.atomic_add(out_full_ptr + out_off, contrib, mask=mask2d)


@triton.jit
def finalize_kernel(
    out_full_ptr, bias_ptr, out_ptr,
    N, OC, OH, OW,
    multiplier,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)  # n * OC + oc
    n = pid // OC
    oc = pid % OC
    HW = OH * OW

    base = pid * HW
    offs = tl.arange(0, BLOCK_HW)
    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    for start in range(0, HW, BLOCK_HW):
        idx = start + offs
        mask = idx < HW
        v = tl.load(out_full_ptr + base + idx, mask=mask, other=0.0)
        acc += v

    s = tl.sum(acc, axis=0)
    b = tl.load(bias_ptr + oc)
    # mean = (s + bias * HW) / HW = s/HW + bias
    mean_val = (s / HW + b) * multiplier
    tl.store(out_ptr + pid, mean_val)


def fused_conv_transpose_mean(x, weight, bias, stride, padding, output_padding, multiplier):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    # Materialize full output tensor (initialized to zero) per safety contract
    out_full = torch.zeros((N, OC, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_HW = 128
    BLOCK_OC = triton.next_power_of_2(OC)
    if BLOCK_OC < 16:
        BLOCK_OC = 16

    grid = (N, IC, triton.cdiv(IH * IW, BLOCK_HW))
    conv_transpose_scatter_kernel[grid](
        x, weight, out_full,
        N, IC, IH, IW, OC, OH, OW,
        KH, KW,
        stride, padding,
        BLOCK_HW=BLOCK_HW,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )

    out = torch.empty((N, OC, 1, 1), device=x.device, dtype=x.dtype)
    BLOCK_HW2 = 1024
    finalize_kernel[(N * OC,)](
        out_full, bias, out,
        N, OC, OH, OW,
        multiplier,
        BLOCK_HW=BLOCK_HW2,
        num_warps=4,
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
        return fused_conv_transpose_mean(
            x, weight, bias,
            self.stride, self.padding, self.output_padding,
            self.multiplier,
        )