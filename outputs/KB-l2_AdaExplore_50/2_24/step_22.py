import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC,
    D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    OC: tl.constexpr, OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N, ceil(OH*OW / BLOCK_HW))
    n = tl.program_id(0)
    hw_block = tl.program_id(1)

    offs_hw = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < (OH * OW)
    oh = offs_hw // OW
    ow = offs_hw % OW

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # bias: (BLOCK_OC,)
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)

    # min accumulator: (BLOCK_OC, BLOCK_HW)
    INF = float('inf')
    min_val = tl.full((BLOCK_OC, BLOCK_HW), INF, dtype=tl.float32)

    # Loop over output depth
    for od in range(0, OD):
        acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)
        for ic in range(0, IC):
            for kd in range(0, KD):
                id_ = od + kd
                for kh in range(0, KH):
                    ih = oh + kh
                    for kw in range(0, KW):
                        iw = ow + kw
                        x_off = ((n * IC + ic) * D + id_) * H * W + ih * W + iw
                        # weight offset: (OC, IC, KD, KH, KW)
                        w_off = ((offs_oc * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                        x_val = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)  # (BLOCK_HW,)
                        w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # (BLOCK_OC,)
                        acc += w_val[:, None] * x_val[None, :]
        acc = acc + bias[:, None]
        min_val = tl.minimum(min_val, acc)

    # softmax over OC axis
    # mask out invalid OC rows with -inf
    min_val = tl.where(mask_oc[:, None], min_val, -INF)
    m = tl.max(min_val, axis=0)  # (BLOCK_HW,)
    e = tl.exp(min_val - m[None, :])
    e = tl.where(mask_oc[:, None], e, 0.0)
    s = tl.sum(e, axis=0)  # (BLOCK_HW,)
    out_val = e / s[None, :]

    # Store: output shape (N, OC, OH, OW)
    out_off = (n * OC + offs_oc[:, None]) * (OH * OW) + offs_hw[None, :]
    mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_off, out_val, mask=mask)


def conv3d_min_softmax(x, weight, bias):
    N, IC, D, H, W = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = D - KD + 1
    OH = H - KH + 1
    OW = W - KW + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_HW = 64
    # next pow2 >= OC
    BLOCK_OC = 1
    while BLOCK_OC < OC:
        BLOCK_OC *= 2

    grid = (N, (OH * OW + BLOCK_HW - 1) // BLOCK_HW)

    conv3d_min_softmax_kernel[grid](
        x, weight, bias, out,
        N, IC,
        D, H, W,
        OC, OD, OH, OW,
        KD, KH, KW,
        BLOCK_HW=BLOCK_HW,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )
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
            return conv3d_min_softmax(x, w, b)
        else:
            x = self.conv(x)
            x = torch.min(x, dim=self.dim)[0]
            x = torch.softmax(x, dim=1)
            return x