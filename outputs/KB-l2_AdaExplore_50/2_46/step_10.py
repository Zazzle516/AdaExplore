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
    KH: tl.constexpr,
    KW: tl.constexpr,
    PK: tl.constexpr,
    SUB1: tl.constexpr,
    SUB2: tl.constexpr,
    INV_PK2: tl.constexpr,
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

    # Output spatial coordinates of the conv (top-left of pooling window)
    oh_base = poh * PK
    ow_base = pow_ * PK

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Iterate over pooling window
    for ph in tl.static_range(0, PK):
        for pw in tl.static_range(0, PK):
            oh = oh_base + ph  # [BLOCK_SP]
            ow = ow_base + pw  # [BLOCK_SP]

            # Compute conv at (oh, ow) for each oc
            conv_val = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

            for ic in range(0, IC):
                for kh in tl.static_range(0, KH):
                    for kw in tl.static_range(0, KW):
                        ih = oh + kh  # [BLOCK_SP]
                        iw = ow + kw  # [BLOCK_SP]

                        # Load input: x[pid_n, ic, ih, iw] -> [BLOCK_SP]
                        x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                        x_mask = sp_mask
                        x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_SP]

                        # Load weight: w[oc, ic, kh, kw] -> [BLOCK_OC]
                        w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                        conv_val += w_val[:, None] * x_val[None, :]

            # Add bias
            b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
            conv_val += b_val[:, None]

            # Subtract sub1, tanh, subtract sub2
            v = conv_val - SUB1
            # tanh via exp
            e2x = tl.exp(2.0 * v)
            t = (e2x - 1.0) / (e2x + 1.0)
            v = t - SUB2

            acc += v

    acc = acc * INV_PK2

    # Store: out[pid_n, oc, poh, pow] - layout [N, OC, POH, POW]
    out_off = pid_n * (OC * POH * POW) + oc_offs[:, None] * (POH * POW) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size_pool = int(kernel_size_pool)
        self.kernel_size = int(kernel_size)
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
        PK = self.kernel_size_pool
        POH = OH // PK
        POW = OW // PK

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 64

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(POH * POW, BLOCK_SP))

        fused_conv_tanh_pool_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW, PK,
            self.subtract1_value, self.subtract2_value,
            1.0 / (PK * PK),
            BLOCK_OC, BLOCK_SP,
            num_warps=4, num_stages=2,
        )

        return out