import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused conv3d (no padding, stride=1) + min over OD + softmax over OC.
# Strategy: one program per (N, HW_tile). It computes for each output (h,w) tile
# the OC outputs (after min over OD), then softmax over OC, then stores.
#
# Layout: input NCDHW, weight (OC, IC, KD, KH, KW). KD=KH=KW=3, OC=24, IC=3.
# OD = D-2, OH = H-2, OW = W-2 = 22, 30, 30.
#
# We use BLOCK_HW threads per output tile. For each tile we accumulate min over OD
# of the conv result, then do softmax across OC=24 in shared registers per element.

@triton.jit
def fused_conv3d_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, D, H, W,
    OC: tl.constexpr, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    hw_block = tl.program_id(1)

    offs = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    OHW = OH * OW
    mask_hw = offs < OHW
    oh = offs // OW
    ow = offs % OW

    # We'll keep min_vals as a 2D tensor (OC, BLOCK_HW)
    INF = float('inf')
    min_vals = tl.full((OC, BLOCK_HW), INF, dtype=tl.float32)

    # Preload biases (OC,)
    oc_range = tl.arange(0, OC)
    bias = tl.load(b_ptr + oc_range)  # (OC,)

    # Loop over OD
    for od in range(0, OD):
        # accumulator (OC, BLOCK_HW)
        acc = tl.zeros((OC, BLOCK_HW), dtype=tl.float32)
        for ic in range(0, IC):
            for kd in range(0, KD):
                id_ = od + kd
                for kh in range(0, KH):
                    ih = oh + kh  # (BLOCK_HW,)
                    for kw in range(0, KW):
                        iw = ow + kw  # (BLOCK_HW,)
                        # Load x[(n, ic, id_, ih, iw)] for BLOCK_HW points
                        x_off = ((n * IC + ic) * D + id_) * H * W + ih * W + iw
                        x_val = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)  # (BLOCK_HW,)
                        # Load weight w[oc, ic, kd, kh, kw] for all OC
                        w_off = ((oc_range * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off)  # (OC,)
                        # outer: (OC, BLOCK_HW)
                        acc += w_val[:, None] * x_val[None, :]
        acc = acc + bias[:, None]
        min_vals = tl.minimum(min_vals, acc)

    # Softmax over OC dim (axis=0)
    m = tl.max(min_vals, axis=0)  # (BLOCK_HW,)
    e = tl.exp(min_vals - m[None, :])
    s = tl.sum(e, axis=0)  # (BLOCK_HW,)
    out = e / s[None, :]

    # Store output (N, OC, OH, OW)
    # out[n, oc, offs]
    out_off = (n * OC + oc_range)[:, None] * OHW + offs[None, :]
    store_mask = mask_hw[None, :]
    tl.store(out_ptr + out_off, out, mask=store_mask)


def fused_conv3d_min_softmax(x, weight, bias):
    N, IC, D, H, W = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = D - KD + 1
    OH = H - KH + 1
    OW = W - KW + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_HW = 64
    OHW = OH * OW
    grid = (N, (OHW + BLOCK_HW - 1) // BLOCK_HW)

    fused_conv3d_min_softmax_kernel[grid](
        x, weight, bias, out,
        N, IC, D, H, W,
        OC, OD, OH, OW,
        KD, KH, KW,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()
        if self.dim == 2:
            return fused_conv3d_min_softmax(x, w, b)
        else:
            x = self.conv(x)
            x = torch.min(x, dim=self.dim)[0]
            x = torch.softmax(x, dim=1)
            return x