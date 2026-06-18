import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N, ceil(OH*OW / BLOCK_HW), ceil(OC / BLOCK_OC))
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_oc = tl.program_id(2)

    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    oh = hw_offs // OW
    ow = hw_offs % OW

    hw_mask = hw_offs < (OH * OW)
    oc_mask = oc_offs < OC

    acc = tl.zeros([BLOCK_HW, BLOCK_OC], dtype=tl.float32)

    # ConvTranspose2d stride=1 equivalent: regular conv with flipped kernel
    # Output[n, oc, oh, ow] = sum_{ic, kh, kw} input[n, ic, oh - kh + pad, ow - kw + pad] * weight[ic, oc, kh, kw]
    # where weight is original ConvTranspose weight (IC, OC, KH, KW)
    for ic in range(0, IC):
        for kh in range(0, KH):
            for kw in range(0, KW):
                ih = oh - kh + PAD
                iw = ow - kw + PAD
                in_mask = hw_mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                in_offs = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                x_val = tl.load(x_ptr + in_offs, mask=in_mask, other=0.0)  # [BLOCK_HW]

                w_offs = ic * OC * KH * KW + oc_offs * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += x_val[:, None] * w_val[None, :]

    # Add bias
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_val[None, :]

    # GELU exact
    acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865475))

    # Store to output [N, OC, OH, OW]
    out_offs = (pid_n * OC * OH * OW
                + oc_offs[None, :] * OH * OW
                + hw_offs[:, None])
    out_mask = hw_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv_transpose_gelu(x, weight, bias, kernel_size):
    N, IC, IH, IW = x.shape
    KH = KW = kernel_size
    OC = weight.shape[1]
    PAD = KH - 1
    OH = IH + 2 * PAD - KH + 1  # = IH + KH - 1
    OW = IW + 2 * PAD - KW + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_HW = 128
    BLOCK_OC = 64

    grid = (N, triton.cdiv(OH * OW, BLOCK_HW), triton.cdiv(OC, BLOCK_OC))

    conv_transpose_gelu_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW, PAD,
        BLOCK_HW=BLOCK_HW,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def groupnorm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    N, C, HW, G, CPG,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_size = CPG * HW
    base = n * C * HW + g * CPG * HW

    sum_val = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)

    num_iters = (group_size + BLOCK - 1) // BLOCK
    for i in range(num_iters):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < group_size
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x, 0.0)
        sum_val += x
        sum_sq += x * x

    total_sum = tl.sum(sum_val, axis=0)
    total_sq = tl.sum(sum_sq, axis=0)
    inv_gs = 1.0 / group_size
    mean = total_sum * inv_gs
    var = total_sq * inv_gs - mean * mean
    rstd = tl.rsqrt(var + eps)

    for i in range(num_iters):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < group_size
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

        c_local = offs // HW
        c_global = g * CPG + c_local
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0)

        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + base + offs, y, mask=mask)


def groupnorm(x, weight, bias, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    CPG = C // num_groups
    out = torch.empty_like(x)

    group_size = CPG * HW
    if group_size >= 8192:
        BLOCK = 2048
        num_warps = 8
    elif group_size >= 4096:
        BLOCK = 1024
        num_warps = 8
    elif group_size >= 1024:
        BLOCK = 512
        num_warps = 4
    else:
        BLOCK = 256
        num_warps = 4

    grid = (N * num_groups,)
    groupnorm_kernel[grid](
        x, out, weight, bias,
        N, C, HW, num_groups, CPG,
        eps,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups
        self.eps = 1e-5
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            x = x.contiguous()
            w = self.conv_transpose.weight.contiguous()
            b = self.conv_transpose.bias.contiguous()
            x = conv_transpose_gelu(x, w, b, self.kernel_size)
        else:
            x = self.conv_transpose(x)
            x = torch.nn.functional.gelu(x)
        x = groupnorm(x, self.group_norm.weight, self.group_norm.bias, self.num_groups, self.eps)
        return x