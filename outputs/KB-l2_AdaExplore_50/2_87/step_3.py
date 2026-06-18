import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_sub_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_IN, H_IN, W_IN,
    C_OUT, H_OUT, W_OUT,
    SUB,
    KH: tl.constexpr, KW: tl.constexpr,
    C_IN_C: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oh = offs_hw // W_OUT
    ow = offs_hw % W_OUT

    mask_oc = offs_oc < C_OUT
    mask_hw = offs_hw < (H_OUT * W_OUT)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # weight shape: [C_OUT, C_IN, KH, KW]
    # input shape: [N, C_IN, H_IN, W_IN]
    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            ih = oh + kh  # [BLOCK_HW]
            iw = ow + kw  # [BLOCK_HW]
            # base x ptr for this (n, kh, kw) over c_in and hw
            for ic in range(0, C_IN_C):
                # load weight [BLOCK_OC]
                w_off = offs_oc * (C_IN * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                x_off = pid_n * (C_IN * H_IN * W_IN) + ic * (H_IN * W_IN) + ih * W_IN + iw
                x_val = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)  # [BLOCK_HW]

                acc += w_val[:, None] * x_val[None, :]

    # add bias
    b_val = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b_val[:, None]

    # subtract
    acc = acc - SUB

    # Mish: x * tanh(softplus(x)) = x * tanh(log(1+exp(x)))
    # Use stable softplus
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = acc * th

    out_off = (pid_n * C_OUT * H_OUT * W_OUT
               + offs_oc[:, None] * (H_OUT * W_OUT)
               + offs_hw[None, :])
    mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_off, out, mask=mask)


def conv2d_sub_mish(x, weight, bias, sub):
    N, C_IN, H_IN, W_IN = x.shape
    C_OUT, _, KH, KW = weight.shape
    H_OUT = H_IN - KH + 1
    W_OUT = W_IN - KW + 1

    out = torch.empty((N, C_OUT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_HW = 128

    grid = (N,
            triton.cdiv(C_OUT, BLOCK_OC),
            triton.cdiv(H_OUT * W_OUT, BLOCK_HW))

    conv2d_sub_mish_kernel[grid](
        x, weight, bias, out,
        N, C_IN, H_IN, W_IN,
        C_OUT, H_OUT, W_OUT,
        sub,
        KH=KH, KW=KW,
        C_IN_C=C_IN,
        BLOCK_OC=BLOCK_OC,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.sub_total = float(subtract_value_1 + subtract_value_2)

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()
        return conv2d_sub_mish(x, w, b, self.sub_total)