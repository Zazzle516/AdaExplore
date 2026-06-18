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
    OHW = OH * OW
    mask_hw = offs < OHW
    oh = offs // OW
    ow = offs % OW

    bias = tl.load(b_ptr + oc)

    # Initialize min to +inf
    min_val = tl.full([BLOCK_HW], float('inf'), dtype=tl.float32)

    # For each output depth position
    for od in range(0, OD):
        acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
        # Convolution sum over IC, KD, KH, KW
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

    # Store output: shape (N, OC, OH, OW)
    out_off = ((n * OC + oc) * OH * OW) + offs
    tl.store(out_ptr + out_off, min_val, mask=mask_hw)


@triton.jit
def softmax_channel_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    # grid: (N, S)
    n = tl.program_id(0)
    s = tl.program_id(1)

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = n * C * S + s
    x_ptrs = x_ptr + base + offs_c * S
    x = tl.load(x_ptrs, mask=mask_c, other=-float('inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask_c, e, 0.0)
    z = tl.sum(e, axis=0)
    y = e / z

    out_ptrs = out_ptr + base + offs_c * S
    tl.store(out_ptrs, y, mask=mask_c)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.dim != 2:
            # fallback
            y = self.conv(x)
            y = torch.min(y, dim=self.dim)[0]
            return torch.softmax(y, dim=1)

        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, D, H, W = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1

        out_min = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_HW = 64
        grid = (N, OC, (OH * OW + BLOCK_HW - 1) // BLOCK_HW)
        conv3d_min_kernel[grid](
            x, w, b, out_min,
            N, IC, D, H, W,
            OC, OD, OH, OW,
            KD, KH, KW,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
        )

        # Softmax along channel dim (dim=1) of (N, OC, OH, OW)
        out = torch.empty_like(out_min)
        S = OH * OW
        BLOCK_C = _next_pow2(OC)
        grid2 = (N, S)
        softmax_channel_kernel[grid2](
            out_min, out,
            N, OC, S,
            BLOCK_C=BLOCK_C,
            num_warps=1,
        )
        return out