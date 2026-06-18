import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_3x3_s1_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_IN, H_IN, W_IN,
    C_OUT, H_OUT, W_OUT,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # grid: (N * C_OUT, ceil(H_OUT/BLOCK_H), ceil(W_OUT/BLOCK_W))
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    n = pid_nc // C_OUT
    oc = pid_nc % C_OUT

    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W

    offs_h = h_start + tl.arange(0, BLOCK_H)  # [BH]
    offs_w = w_start + tl.arange(0, BLOCK_W)  # [BW]

    out_mask = (offs_h[:, None] < H_OUT) & (offs_w[None, :] < W_OUT)

    bias_val = tl.load(b_ptr + oc)
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32) + bias_val

    # ConvTranspose 3x3 stride=1 no padding equivalent:
    # y[oh, ow] = sum_{kh, kw, ic} x[oh - kh, ow - kw, ic] * w[ic, oc, kh, kw]
    # where valid input (oh-kh) in [0, H_IN), (ow-kw) in [0, W_IN)
    # Equivalent to full-conv with flipped kernel, padding=2 on input.

    for ic in range(0, C_IN):
        for kh in tl.static_range(0, 3):
            for kw in tl.static_range(0, 3):
                ih = offs_h[:, None] - kh  # [BH, 1]
                iw = offs_w[None, :] - kw  # [1, BW]
                in_mask = (ih >= 0) & (ih < H_IN) & (iw >= 0) & (iw < W_IN) & out_mask
                ih_c = tl.where(in_mask, ih, 0)
                iw_c = tl.where(in_mask, iw, 0)
                x_off = ((n * C_IN + ic) * H_IN + ih_c) * W_IN + iw_c
                xv = tl.load(x_ptr + x_off, mask=in_mask, other=0.0)
                # weight layout: (C_IN, C_OUT, 3, 3) -> idx = ((ic * C_OUT + oc) * 3 + kh) * 3 + kw
                w_off = ((ic * C_OUT + oc) * 3 + kh) * 3 + kw
                wv = tl.load(w_ptr + w_off)
                acc += xv * wv

    y_off = ((n * C_OUT + oc) * H_OUT + offs_h[:, None]) * W_OUT + offs_w[None, :]
    tl.store(y_ptr + y_off, acc, mask=out_mask)


def conv_transpose_3x3_s1(x, weight, bias):
    N, C_IN, H_IN, W_IN = x.shape
    C_OUT = weight.shape[1]
    H_OUT = H_IN + 2
    W_OUT = W_IN + 2

    y = torch.empty((N, C_OUT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    BLOCK_H = 16
    BLOCK_W = 64

    grid = (N * C_OUT, triton.cdiv(H_OUT, BLOCK_H), triton.cdiv(W_OUT, BLOCK_W))
    conv_transpose_3x3_s1_kernel[grid](
        x, weight, bias, y,
        N, C_IN, H_IN, W_IN,
        C_OUT, H_OUT, W_OUT,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=4, num_stages=2,
    )
    return y


@triton.jit
def gelu_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    HW, GROUP_SIZE, NUM_GROUPS, C,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    group_elems = GROUP_SIZE * HW
    base = n * C * HW + g * GROUP_SIZE * HW

    inv_sqrt2 = 0.7071067811865475

    sum_x = 0.0
    sum_x2 = 0.0

    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        gelu = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
        gelu = tl.where(mask, gelu, 0.0)
        sum_x += tl.sum(gelu)
        sum_x2 += tl.sum(gelu * gelu)

    inv_n = 1.0 / group_elems
    mean = sum_x * inv_n
    var = sum_x2 * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        gelu = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

        c_local = idx // HW
        c_global = g * GROUP_SIZE + c_local
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0)

        out = (gelu - mean) * rstd * w + b
        tl.store(y_ptr + base + idx, out, mask=mask)


def fused_gelu_groupnorm(x, weight, bias, num_groups, eps=1e-5):
    assert x.is_cuda and x.is_contiguous()
    N, C, H, W = x.shape
    HW = H * W
    GROUP_SIZE = C // num_groups
    y = torch.empty_like(x)

    grid = (N * num_groups,)

    BLOCK_SIZE = 4096
    num_warps = 8
    num_stages = 3

    gelu_groupnorm_kernel[grid](
        x, y, weight, bias,
        HW, GROUP_SIZE, num_groups, C,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


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
        x = x.contiguous()
        if self.kernel_size == 3 and self.stride == 1:
            w = self.conv_transpose.weight.contiguous()
            b = self.conv_transpose.bias.contiguous()
            x = conv_transpose_3x3_s1(x, w, b)
        else:
            x = self.conv_transpose(x)
        x = x.contiguous()
        y = fused_gelu_groupnorm(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.num_groups,
            self.eps,
        )
        return y