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
    BLOCK_IC: tl.constexpr,
):
    # x is NHWC: (N, IH, IW, IC)
    # weight pre-packed as (KH*KW*IC, OC) for contiguous K-dim GEMM
    # out NHWC pooled: (N, POH*POW, OC)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (POH * POW)

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    # base for top-left of pool window in conv coords
    oh_base = poh * POOL  # [BLOCK_SP]
    ow_base = pow_ * POOL  # [BLOCK_SP]

    K = KH * KW * IC  # K dimension length

    # accumulators for each pool sub-position separately
    # We'll store partial conv results in a single accumulator that gets
    # the tanh applied per-subpixel; so we need POOL*POOL accumulators.
    # For POOL=2, that's 4 acc tiles. We handle via static loop.

    pooled = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # bias pre-load
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

    inv_pool = 1.0 / (POOL * POOL)

    for ph in tl.static_range(0, POOL):
        for pw in tl.static_range(0, POOL):
            oh = oh_base + ph  # [BLOCK_SP]
            ow = ow_base + pw  # [BLOCK_SP]

            conv_val = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

            # Loop over K = KH*KW*IC in chunks of BLOCK_IC, but we step through
            # (kh, kw, ic_block). Use static loops on kh, kw and dynamic on ic.
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    ih = oh + kh  # [BLOCK_SP]
                    iw = ow + kw  # [BLOCK_SP]
                    # x base offsets [BLOCK_SP] without ic
                    x_base = (pid_n * IH * IW * IC
                              + ih * (IW * IC)
                              + iw * IC)  # [BLOCK_SP]
                    # weight K-row base for this (kh, kw)
                    k_row_base = (kh * KW + kw) * IC  # scalar

                    for ic_base in range(0, IC, BLOCK_IC):
                        ic_offs = ic_base + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
                        ic_mask = ic_offs < IC

                        # x: [BLOCK_SP, BLOCK_IC]
                        x_off = x_base[:, None] + ic_offs[None, :]
                        x_mask = sp_mask[:, None] & ic_mask[None, :]
                        x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                        # w: [BLOCK_IC, BLOCK_OC] from packed (K, OC)
                        k_offs = k_row_base + ic_offs  # [BLOCK_IC]
                        w_off = k_offs[:, None] * OC + oc_offs[None, :]
                        w_mask = ic_mask[:, None] & oc_mask[None, :]
                        w_val = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                        conv_val += tl.dot(x_val, w_val)

            conv_val += b_val[None, :]
            v = conv_val - SUB1
            # tanh
            e2 = tl.exp(2.0 * v)
            t = (e2 - 1.0) / (e2 + 1.0)
            v = t - SUB2
            pooled += v

    pooled = pooled * inv_pool

    out_off = (pid_n * (POH * POW * OC)
               + sp_offs[:, None] * OC
               + oc_offs[None, :])
    mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, pooled, mask=mask)


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

        # Pre-pack weight to (KH*KW*IC, OC) layout
        w = self.conv.weight.detach().contiguous()  # (OC, IC, KH, KW)
        OC, IC, KH, KW = w.shape
        # rearrange to (KH, KW, IC, OC) then flatten to (KH*KW*IC, OC)
        w_packed = w.permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC).contiguous()
        self.register_buffer("w_packed", w_packed.cuda())
        self.register_buffer("b_buf", self.conv.bias.detach().contiguous().cuda())

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, IH, IW, IC)

        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.kernel_size_pool
        POH = OH // POOL
        POW = OW // POOL

        out_nhwc = torch.empty((N, POH * POW, OC), device=x.device, dtype=torch.float32)

        BLOCK_OC = 64
        BLOCK_SP = 64
        BLOCK_IC = 32

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(POH * POW, BLOCK_SP))

        fused_conv_tanh_pool_kernel[grid](
            x_nhwc, self.w_packed, self.b_buf, out_nhwc,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.subtract1_value, self.subtract2_value,
            BLOCK_OC, BLOCK_SP, BLOCK_IC,
            num_warps=4, num_stages=3,
        )

        out = out_nhwc.view(N, POH, POW, OC).permute(0, 3, 1, 2).contiguous()
        return out