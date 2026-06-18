import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_HW': 512}, num_warps=8, num_stages=2),
    ],
    key=['C_OUT', 'HW_OUT', 'C_IN_C'],
)
@triton.jit
def conv2d_sub_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H_IN, W_IN,
    C_OUT, H_OUT, W_OUT, HW_OUT,
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
    mask_hw = offs_hw < HW_OUT

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    x_batch_base = pid_n * (C_IN_C * H_IN * W_IN)

    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            ih = oh + kh
            iw = ow + kw
            hw_off = ih * W_IN + iw
            for ic in tl.static_range(C_IN_C):
                w_off = offs_oc * (C_IN_C * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)

                x_off = x_batch_base + ic * (H_IN * W_IN) + hw_off
                x_val = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)

                acc += w_val[:, None] * x_val[None, :]

    b_val = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b_val[:, None] - SUB

    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = acc * th

    out_off = (pid_n * C_OUT * HW_OUT
               + offs_oc[:, None] * HW_OUT
               + offs_hw[None, :])
    mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_off, out, mask=mask)


def conv2d_sub_mish(x, weight, bias, sub):
    N, C_IN, H_IN, W_IN = x.shape
    C_OUT, _, KH, KW = weight.shape
    H_OUT = H_IN - KH + 1
    W_OUT = W_IN - KW + 1
    HW_OUT = H_OUT * W_OUT

    out = torch.empty((N, C_OUT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    grid = lambda meta: (N,
                         triton.cdiv(C_OUT, meta['BLOCK_OC']),
                         triton.cdiv(HW_OUT, meta['BLOCK_HW']))

    conv2d_sub_mish_kernel[grid](
        x, weight, bias, out,
        N, H_IN, W_IN,
        C_OUT, H_OUT, W_OUT, HW_OUT,
        sub,
        KH=KH, KW=KW,
        C_IN_C=C_IN,
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