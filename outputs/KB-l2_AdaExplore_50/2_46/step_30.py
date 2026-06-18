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

    # The pooled output is averaged over POOL x POOL conv outputs starting at (poh*POOL, pow_*POOL)
    # We accumulate sum of tanh(conv - sub1) - sub2 over the pool window, then divide by POOL*POOL.

    pool_inv = 1.0 / (POOL * POOL)

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Load bias for these output channels
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

    # Loop over pool window
    for ph in tl.static_range(POOL):
        for pw in tl.static_range(POOL):
            oh = poh * POOL + ph  # [BLOCK_SP]
            ow = pow_ * POOL + pw  # [BLOCK_SP]

            # conv at (oh, ow) for output channels oc_offs
            # conv[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, oh+kh, ow+kw] * w[oc, ic, kh, kw] + b[oc]
            conv_acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

            for ic in range(0, IC):
                for kh in tl.static_range(KH):
                    for kw in tl.static_range(KW):
                        ih = oh + kh  # [BLOCK_SP]
                        iw = ow + kw  # [BLOCK_SP]
                        # Load input [BLOCK_SP]
                        x_idx = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                        x_val = tl.load(x_ptr + x_idx, mask=sp_mask, other=0.0)  # [BLOCK_SP]

                        # Load weight [BLOCK_OC]
                        w_idx = oc_offs * IC * KH * KW + ic * KH * KW + kh * KW + kw
                        w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                        conv_acc += w_val[:, None] * x_val[None, :]

            conv_acc = conv_acc + bias[:, None]
            # Apply: tanh(conv - sub1) - sub2
            v = conv_acc - SUB1
            # tanh via sigmoid: tanh(x) = 2*sigmoid(2x) - 1
            t = 2.0 * tl.sigmoid(2.0 * v) - 1.0
            t = t - SUB2
            acc += t * pool_inv

    # Store
    out_idx = pid_n * OC * POH * POW + oc_offs[:, None] * POH * POW + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size_pool = kernel_size_pool
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

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

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

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