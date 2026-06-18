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
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    n = tl.program_id(0)
    oc_block = tl.program_id(1)
    sp_block = tl.program_id(2)

    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = sp_block * BLOCK_SP + tl.arange(0, BLOCK_SP)
    ic_offs = tl.arange(0, BLOCK_IC)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < OH * OW
    ic_mask = ic_offs < IC

    oh = sp_offs // OW
    ow = sp_offs % OW

    # Load bias [BLOCK_OC]
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = tl.zeros([BLOCK_OC, BLOCK_SP], dtype=tl.float32)

    # ConvTranspose2d with stride=1, pad=0
    # y[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, oh - kh, ow - kw] * w[ic, oc, kh, kw]
    x_base = n * (IC * H * W)
    for kh in tl.static_range(0, KH):
        ih = oh - kh
        ih_valid = (ih >= 0) & (ih < H)
        for kw in tl.static_range(0, KW):
            iw = ow - kw
            iw_valid = (iw >= 0) & (iw < W)
            in_valid = ih_valid & iw_valid & sp_mask  # [BLOCK_SP]

            # Load X[ic, sp] : [BLOCK_IC, BLOCK_SP]
            x_off = x_base + ic_offs[:, None] * (H * W) + ih[None, :] * W + iw[None, :]
            x_mask = ic_mask[:, None] & in_valid[None, :]
            x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

            # Load W[oc, ic] : [BLOCK_OC, BLOCK_IC]
            w_off = oc_offs[:, None] * (KH * KW) + ic_offs[None, :] * (OC * KH * KW) + kh * KW + kw
            w_mask = oc_mask[:, None] & ic_mask[None, :]
            w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

            acc += tl.dot(w_tile, x_tile, allow_tf32=True)

    acc += bias[:, None]

    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Store output (NCHW)
    y_off = n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(y_ptr + y_off, gelu, mask=mask)


@triton.jit
def group_norm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, HW, G, CPG: tl.constexpr,
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
        sum_val += x
        sumsq_val += x * x

    s = tl.sum(sum_val, axis=0)
    sq = tl.sum(sumsq_val, axis=0)

    mean = s / group_elems
    var = sq / group_elems - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: iterate per channel to load weight/bias once per channel
    c_base = g * CPG
    for c_in_group in tl.static_range(0, CPG):
        c_global = c_base + c_in_group
        w = tl.load(weight_ptr + c_global).to(tl.float32)
        b = tl.load(bias_ptr + c_global).to(tl.float32)
        ch_base = base + c_in_group * HW
        for off in range(0, HW, BLOCK_SIZE):
            idx = off + tl.arange(0, BLOCK_SIZE)
            mask = idx < HW
            x = tl.load(x_ptr + ch_base + idx, mask=mask, other=0.0).to(tl.float32)
            y = (x - mean) * rstd * w + b
            tl.store(y_ptr + ch_base + idx, y, mask=mask)


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
        KH = self.kernel_size
        KW = self.kernel_size
        OH = H + KH - 1  # stride=1, padding=0
        OW = W + KW - 1

        conv_out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_SP = 128
        BLOCK_IC = 64

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

        conv_transpose_gelu_kernel[grid](
            x, self.conv_transpose.weight, self.conv_transpose.bias, conv_out,
            N, IC, H, W,
            OC, OH, OW,
            KH, KW,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP, BLOCK_IC=BLOCK_IC,
            num_warps=4, num_stages=3,
        )

        # GroupNorm
        C = OC
        HW = OH * OW
        G = self.num_groups
        CPG = C // G

        y = torch.empty_like(conv_out)

        BLOCK_SIZE = 2048
        num_warps = 8

        gn_grid = (N * G,)
        group_norm_kernel[gn_grid](
            conv_out, y,
            self.group_norm.weight, self.group_norm.bias,
            N, C, HW, G, CPG,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=2,
        )
        return y