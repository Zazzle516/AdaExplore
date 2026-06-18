import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N, OC, ceil(OH*OW / BLOCK_HW))
    n = tl.program_id(0)
    oc = tl.program_id(1)
    hw_block = tl.program_id(2)

    offs = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs < (OH * OW)
    oh = offs // OW
    ow = offs % OW

    bias = tl.load(b_ptr + oc)

    # Initialize min accumulator
    INF = float('inf')
    min_val = tl.full((BLOCK_HW,), INF, dtype=tl.float32)

    # Loop over output depth
    for od in range(0, OD):
        acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
        # Loop over input channels and kernel
        for ic in range(0, IC):
            for kd in range(0, KD):
                id_ = od + kd
                for kh in range(0, KH):
                    ih = oh + kh
                    for kw in range(0, KW):
                        iw = ow + kw
                        x_off = ((n * IC + ic) * D + id_) * H * W + ih * W + iw
                        w_off = ((oc * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                        x_val = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)
                        w_val = tl.load(w_ptr + w_off)
                        acc += x_val * w_val
        acc = acc + bias
        min_val = tl.minimum(min_val, acc)

    # Store result: shape (N, OC, OH, OW)
    out_off = ((n * OC + oc) * OH * OW) + offs
    tl.store(out_ptr + out_off, min_val, mask=mask_hw)


@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    # grid: (N, S)
    n = tl.program_id(0)
    s = tl.program_id(1)
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C

    base = n * C * S + s
    ptrs = x_ptr + offs_c * S + base
    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s_sum = tl.sum(e, axis=0)
    out = e / s_sum
    out_ptrs = out_ptr + offs_c * S + base
    tl.store(out_ptrs, out, mask=mask)


def conv3d_min_dim2(x, weight, bias):
    N, IC, D, H, W = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = D - KD + 1
    OH = H - KH + 1
    OW = W - KW + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_HW = 128
    grid = (N, OC, (OH * OW + BLOCK_HW - 1) // BLOCK_HW)

    conv3d_min_kernel[grid](
        x, weight, bias, out,
        N, IC, D, H, W,
        OC, OD, OH, OW,
        KD, KH, KW,
        BLOCK_HW=BLOCK_HW,
    )
    return out


def softmax_dim1(x):
    N, C, H, W = x.shape
    S = H * W
    out = torch.empty_like(x)
    # Choose BLOCK_C as next pow2 >= C
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2
    grid = (N, S)
    softmax_kernel[grid](x, out, N, C, S, BLOCK_C=BLOCK_C)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()
        if self.dim == 2:
            y = conv3d_min_dim2(x, w, b)
            y = softmax_dim1(y)
            return y
        else:
            x = self.conv(x)
            x = torch.min(x, dim=self.dim)[0]
            x = torch.softmax(x, dim=1)
            return x