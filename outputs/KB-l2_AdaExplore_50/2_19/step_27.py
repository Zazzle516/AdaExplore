import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_gelu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # Implements ConvTranspose2d with stride=1, kernel KHxKW as a Conv2d on padded
    # input with a flipped weight: y[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, oh - kh + PAD_in, ow - kw + PAD_in]_pad * w[ic, oc, kh, kw]
    # For stride=1 and ConvTranspose2d: y[n, oc, oh, ow] = sum x[n, ic, ih, iw] * w[ic, oc, oh-ih, ow-iw], where 0 <= oh-ih < KH
    # Equivalently with padded input by (KH-1, KW-1): y[n, oc, oh, ow] = sum_{ic, kh, kw} x_pad[n, ic, oh+kh, ow+kw] * w[ic, oc, KH-1-kh, KW-1-kw]
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    oh = sp_offs // OW
    ow = sp_offs % OW

    acc = tl.zeros([BLOCK_OC, BLOCK_SP], dtype=tl.float32)

    # Loop over input channels and kernel positions
    for ic in range(0, IC):
        for kh in range(0, KH):
            for kw in range(0, KW):
                # input spatial position: ih = oh + kh - PAD, iw = ow + kw - PAD
                # PAD = KH - 1 (full padding) so that ih = oh + kh - (KH-1)
                ih = oh + kh - PAD
                iw = ow + kw - PAD
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & sp_mask
                x_idx = pid_n * IC * H * W + ic * H * W + ih * W + iw
                x_val = tl.load(x_ptr + x_idx, mask=in_bounds, other=0.0)  # [BLOCK_SP]

                # weight: original w has shape (IC, OC, KH, KW). For transposed conv with flipped kernel,
                # we want w_flipped[ic, oc, kh, kw] = w[ic, oc, KH-1-kh, KW-1-kw]
                fkh = KH - 1 - kh
                fkw = KW - 1 - kw
                w_idx = ic * OC * KH * KW + oc_offs * KH * KW + fkh * KW + fkw
                w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += w_val[:, None] * x_val[None, :]

    # Add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    # GELU (exact)
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Store: y[n, oc, oh, ow]
    out_idx = pid_n * OC * OH * OW + oc_offs[:, None] * (OH * OW) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(y_ptr + out_idx, acc, mask=out_mask)


@triton.jit
def group_norm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, HW, G, CPG,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_elems = CPG * HW
    base = n * C * HW + g * CPG * HW

    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sumsq_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        x_masked = tl.where(mask, x, 0.0)
        sum_val += x_masked
        sumsq_val += x_masked * x_masked

    s = tl.sum(sum_val, axis=0)
    sq = tl.sum(sumsq_val, axis=0)

    mean = s / group_elems
    var = sq / group_elems - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)

        c_in_group = idx // HW
        c_global = g * CPG + c_in_group
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * rstd * w + b
        tl.store(y_ptr + base + idx, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        # ConvTranspose2d with stride=1, no padding: OH = H + KH - 1
        OH = H + KH - 1
        OW = W + KW - 1
        PAD = KH - 1

        # Conv weight has shape (IC, OC, KH, KW)
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()

        conv_out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64 if OC >= 64 else 32
        BLOCK_SP = 128

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))
        conv_transpose_gelu_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, H, W,
            OC, OH, OW,
            KH, KW,
            PAD,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm
        G = self.num_groups
        CPG = OC // G
        HW = OH * OW
        y = torch.empty_like(conv_out)

        group_elems = CPG * HW
        if group_elems >= 4096:
            BLOCK_SIZE = 1024
            num_warps = 8
        elif group_elems >= 1024:
            BLOCK_SIZE = 512
            num_warps = 4
        else:
            BLOCK_SIZE = 256
            num_warps = 4

        gn_grid = (N * G,)
        group_norm_kernel[gn_grid](
            conv_out, y,
            self.group_norm.weight, self.group_norm.bias,
            N, OC, HW, G, CPG,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return y