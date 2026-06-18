import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def _conv_gelu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_IN, H_IN, W_IN,
    C_OUT, H_OUT, W_OUT,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, oc_block, hw_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    HW_OUT = H_OUT * W_OUT
    hw_off = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    h_out = hw_off // W_OUT
    w_out = hw_off % W_OUT
    hw_mask = hw_off < HW_OUT
    oc_mask = oc_off < C_OUT

    # output is conv_transpose with k=3, s=1, p=0 -> equivalent to conv2d on padded input with flipped kernel
    # output_h = input_h + 2 (with padding=1 effectively on padded input of size H_IN+2)
    # we directly compute conv_transpose: out[n,oc,oh,ow] = sum_{ic,kh,kw} x[n,ic,oh-kh,ow-kw] * w[ic,oc,kh,kw]
    # where indices must be valid (0 <= oh-kh < H_IN, 0 <= ow-kw < W_IN)

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    # Loop over kh, kw (3x3)
    for kh in tl.static_range(0, 3):
        ih = h_out - kh  # input row
        ih_valid = (ih >= 0) & (ih < H_IN)
        for kw in tl.static_range(0, 3):
            iw = w_out - kw
            iw_valid = (iw >= 0) & (iw < W_IN)
            valid = hw_mask & ih_valid & iw_valid
            # Clamp index for safe load
            ih_c = tl.where(ih_valid, ih, 0)
            iw_c = tl.where(iw_valid, iw, 0)

            # Loop over input channels
            for ic in range(0, C_IN):
                # x[n, ic, ih, iw]
                x_off = ((pid_n * C_IN + ic) * H_IN + ih_c) * W_IN + iw_c
                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)  # [BLOCK_HW]
                # w[ic, oc, kh, kw]
                w_off = ((ic * C_OUT + oc_off) * 3 + kh) * 3 + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                acc += x_val[:, None] * w_val[None, :]

    # bias
    b_val = tl.load(b_ptr + oc_off, mask=oc_mask, other=0.0)
    acc += b_val[None, :]

    # GELU exact
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865475))

    # store: y[n, oc, h_out, w_out]
    y_off = ((pid_n * C_OUT + oc_off[None, :]) * H_OUT + h_out[:, None]) * W_OUT + w_out[:, None]
    mask = hw_mask[:, None] & oc_mask[None, :]
    tl.store(y_ptr + y_off, gelu, mask=mask)


def conv_transpose_gelu(x, weight, bias):
    # x: [N, C_IN, H_IN, W_IN]
    # weight: [C_IN, C_OUT, 3, 3]  (conv_transpose layout)
    # bias: [C_OUT]
    N, C_IN, H_IN, W_IN = x.shape
    C_OUT = weight.shape[1]
    H_OUT = H_IN + 2
    W_OUT = W_IN + 2

    y = torch.empty((N, C_OUT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    BLOCK_HW = 128
    BLOCK_OC = 64

    grid = (N, triton.cdiv(C_OUT, BLOCK_OC), triton.cdiv(H_OUT * W_OUT, BLOCK_HW))
    _conv_gelu_kernel[grid](
        x, weight, bias, y,
        N, C_IN, H_IN, W_IN,
        C_OUT, H_OUT, W_OUT,
        BLOCK_HW=BLOCK_HW,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )
    return y


@triton.jit
def _groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, HW, GROUP_SIZE, NUM_GROUPS,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    group_elems = GROUP_SIZE * HW
    base = n * C * HW + g * GROUP_SIZE * HW

    sum_val = 0.0
    sum_sq = 0.0

    for off in tl.range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x, 0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / group_elems
    var = sum_sq / group_elems - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in tl.range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        c_in_group = idx // HW
        c_global = g * GROUP_SIZE + c_in_group
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(y_ptr + base + idx, out, mask=mask)


def groupnorm(x, weight, bias, num_groups, eps=1e-5):
    assert x.is_cuda and x.is_contiguous()
    N, C, H, W = x.shape
    HW = H * W
    GROUP_SIZE = C // num_groups
    y = torch.empty_like(x)

    grid = (N * num_groups,)
    BLOCK = 1024
    _groupnorm_kernel[grid](
        x, y, weight, bias,
        N, C, HW, GROUP_SIZE, num_groups,
        eps,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = x.contiguous()
        # Custom conv_transpose with k=3, s=1, p=0 fused with GELU
        if self.kernel_size == 3 and self.stride == 1:
            y = conv_transpose_gelu(
                x,
                self.conv_transpose.weight,
                self.conv_transpose.bias,
            )
        else:
            y = self.conv_transpose(x)
            y = F.gelu(y)
        y = y.contiguous()
        out = groupnorm(
            y,
            self.group_norm.weight,
            self.group_norm.bias,
            self.num_groups,
            self.group_norm.eps,
        )
        return out