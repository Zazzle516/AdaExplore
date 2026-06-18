import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_gelu_avgpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, OC: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offs = pid_s * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    OHOW = OH * OW
    s_mask = s_offs < OHOW

    oh = s_offs // OW
    ow = s_offs % OW

    acc = tl.zeros((BLOCK_SPATIAL,), dtype=tl.float32)

    x_base = pid_n * IC * H * W
    w_base = pid_oc * IC * KH * KW

    for ic in tl.static_range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh
                iw = ow + kw
                x_idx = x_base + ic * H * W + ih * W + iw
                w_val = tl.load(w_ptr + w_base + ic * KH * KW + kh * KW + kw)
                x_val = tl.load(x_ptr + x_idx, mask=s_mask, other=0.0)
                acc += x_val * w_val

    bias = tl.load(b_ptr + pid_oc)
    acc = acc + bias

    # GELU (erf-based exact)
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    gelu = tl.where(s_mask, gelu, 0.0)
    partial_sum = tl.sum(gelu, axis=0)

    scaled = partial_sum / (OH * OW)
    tl.atomic_add(out_ptr + pid_n * OC + pid_oc, scaled)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        weight = self.conv.weight.cuda().contiguous()
        bias = self.conv.bias.cuda().contiguous()

        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_SPATIAL = 256
        OHOW = OH * OW
        grid = (N, OC, (OHOW + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL)

        conv_gelu_avgpool_kernel[grid](
            x, weight, bias, out,
            N, IC, OC, H, W, OH, OW, KH, KW,
            BLOCK_SPATIAL=BLOCK_SPATIAL,
            num_warps=4, num_stages=2,
        )

        return out