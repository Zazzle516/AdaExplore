import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    POH, POW,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SUB1: tl.constexpr, SUB2: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (POH * POW)

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    # output spatial start (top-left of pool window in conv output)
    oh_start = poh * POOL
    ow_start = pow_ * POOL

    # accumulator: [BLOCK_OC, BLOCK_SP] holds the pooled sum
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # iterate over pool window
    for ph in tl.static_range(0, POOL):
        for pw in tl.static_range(0, POOL):
            oh = oh_start + ph  # [BLOCK_SP]
            ow = ow_start + pw  # [BLOCK_SP]

            # compute conv output at (n=pid_n, oc=oc_offs, oh, ow)
            conv_val = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

            for ic in range(0, IC):
                for kh in tl.static_range(0, KH):
                    for kw in tl.static_range(0, KW):
                        ih = oh + kh  # [BLOCK_SP]
                        iw = ow + kw  # [BLOCK_SP]
                        # x[pid_n, ic, ih, iw]
                        x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                        x_val = tl.load(x_ptr + x_off, mask=sp_mask, other=0.0)  # [BLOCK_SP]
                        # w[oc, ic, kh, kw]
                        w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                        conv_val += w_val[:, None] * x_val[None, :]

            # add bias
            b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
            conv_val += b_val[:, None]

            # subtract1, tanh, subtract2
            v = conv_val - SUB1
            # tanh via expression
            e2 = tl.exp(2.0 * v)
            t = (e2 - 1.0) / (e2 + 1.0)
            v = t - SUB2

            acc += v

    acc = acc / (POOL * POOL)

    # store
    out_off = pid_n * (OC * POH * POW) + oc_offs[:, None] * (POH * POW) + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size_pool = int(kernel_size_pool)
        self.kernel_size = int(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.kernel_size_pool
        POH = OH // POOL
        POW = OW // POOL

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)

        BLOCK_OC = 32
        BLOCK_SP = 64

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(POH * POW, BLOCK_SP))

        fused_conv_tanh_pool_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.subtract1_value, self.subtract2_value,
            BLOCK_OC, BLOCK_SP,
            num_warps=4, num_stages=2,
        )
        return out