import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # Grid: (N, ceil(OC/BLOCK_OC), ceil(OH*OW/BLOCK_SP))
    n = tl.program_id(0)
    oc_block = tl.program_id(1)
    sp_block = tl.program_id(2)

    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = sp_block * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < OH * OW

    oh = sp_offs // OW
    ow = sp_offs % OW

    acc = tl.zeros([BLOCK_OC, BLOCK_SP], dtype=tl.float32)

    # For ConvTranspose2d with stride=1, equivalent to Conv2d with flipped kernel and padding=KH-1
    # output[oc, oh, ow] = sum_{ic, kh, kw} input[ic, oh-PAD+kh, ow-PAD+kw] * W_flip[oc, ic, kh, kw]
    # where W_flip[oc, ic, kh, kw] = ConvT.weight[ic, oc, KH-1-kh, KW-1-kw]
    # ConvT.weight has shape [IC, OC, KH, KW]

    for ic in range(IC):
        for kh in range(KH):
            for kw in range(KW):
                ih = oh - PAD + kh
                iw = ow - PAD + kw
                in_mask = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & sp_mask
                x_offs = n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                x_vals = tl.load(x_ptr + x_offs, mask=in_mask, other=0.0)  # [BLOCK_SP]

                # weight: W[ic, oc, KH-1-kh, KW-1-kw]
                w_offs = ic * OC * KH * KW + oc_offs * KH * KW + (KH - 1 - kh) * KW + (KW - 1 - kw)
                w_vals = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += w_vals[:, None] * x_vals[None, :]

    # add bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc += b_vals[:, None]

    # GELU exact
    acc = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865475))

    # store: out[n, oc, oh, ow]
    out_offs = n * OC * OH * OW + oc_offs[:, None] * OH * OW + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv_transpose_gelu(x, weight, bias, kernel_size):
    N, IC, IH, IW = x.shape
    OC = weight.shape[1]
    KH = KW = kernel_size
    PAD = KH - 1
    OH = IH + KH - 1
    OW = IW + KW - 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 64
    BLOCK_SP = 64

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))
    conv_gelu_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        PAD,
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def groupnorm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    C, HW, G, CPG,
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
        C, HW, num_groups, CPG,
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
            x = F.gelu(x)
        x = groupnorm(x, self.group_norm.weight, self.group_norm.bias, self.num_groups, self.eps)
        return x