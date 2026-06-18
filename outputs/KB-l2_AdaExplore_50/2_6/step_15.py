import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, OC: tl.constexpr,
    D, H, W,
    OD_conv, OH_conv, OW_conv,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ow_blocks = (OW + BLOCK_OW - 1) // BLOCK_OW
    owb = pid % num_ow_blocks
    tmp = pid // num_ow_blocks
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    ow_offs = owb * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offs < OW

    # conv output spatial base for the pool window
    d0 = od * POOL
    h0 = oh * POOL

    c_offs = tl.arange(0, OC)  # [OC]

    # constants
    HW = H * W
    DHW = D * HW
    n_base = n * IC * DHW

    max_val = tl.zeros([OC, BLOCK_OW], dtype=tl.float32) - float('inf')

    # iterate over pooling window (POOL^3 conv outputs)
    for dd in tl.static_range(0, POOL):
        od_c = d0 + dd
        for hh in tl.static_range(0, POOL):
            oh_c = h0 + hh
            for ww in tl.static_range(0, POOL):
                ow_c = ow_offs * POOL + ww  # [BLOCK_OW]

                # bias init [OC, BLOCK_OW]
                bias = tl.load(b_ptr + c_offs)  # [OC]
                acc = bias[:, None] + tl.zeros([OC, BLOCK_OW], dtype=tl.float32)

                # convolution
                for kd in tl.static_range(0, KD):
                    d_in = od_c + kd
                    for kh in tl.static_range(0, KH):
                        h_in = oh_c + kh
                        for kw in tl.static_range(0, KW):
                            w_in = ow_c + kw  # [BLOCK_OW]
                            for ic in tl.static_range(0, IC):
                                x_idx = n_base + ic * DHW + d_in * HW + h_in * W + w_in
                                xv = tl.load(x_ptr + x_idx, mask=ow_mask, other=0.0)  # [BLOCK_OW]
                                # weight: [OC, IC, KD, KH, KW]
                                w_idx = c_offs * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                                wv = tl.load(w_ptr + w_idx)  # [OC]
                                acc += wv[:, None] * xv[None, :]

                # softmax across OC (axis=0)
                m = tl.max(acc, axis=0)  # [BLOCK_OW]
                ex = tl.exp(acc - m[None, :])
                s = tl.sum(ex, axis=0)  # [BLOCK_OW]
                sm = ex / s[None, :]
                sm = tl.where(ow_mask[None, :], sm, -float('inf'))
                max_val = tl.maximum(max_val, sm)

    # store [OC, BLOCK_OW] to output [N, OC, OD, OH, OW]
    ODOHOW = OD * OH * OW
    out_base = n * OC * ODOHOW + od * OH * OW + oh * OW
    out_ptrs = out_ptr + out_base + ow_offs[None, :] + c_offs[:, None] * ODOHOW
    tl.store(out_ptrs, max_val, mask=ow_mask[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.pool_total = pool_kernel_size * pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, D, H, W = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        OD_conv = D - KD + 1
        OH_conv = H - KH + 1
        OW_conv = W - KW + 1
        POOL = self.pool_total
        OD = OD_conv // POOL
        OH = OH_conv // POOL
        OW = OW_conv // POOL

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        BLOCK_OW = triton.next_power_of_2(OW)
        if BLOCK_OW < 8:
            BLOCK_OW = 8

        num_ow_blocks = (OW + BLOCK_OW - 1) // BLOCK_OW
        grid = (N * OD * OH * num_ow_blocks,)

        fused_conv_softmax_pool_kernel[grid](
            x, weight, bias, out,
            N, IC, OC,
            D, H, W,
            OD_conv, OH_conv, OW_conv,
            OD, OH, OW,
            KD, KH, KW,
            POOL,
            BLOCK_OW,
            num_warps=4,
            num_stages=2,
        )
        return out