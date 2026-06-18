import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_div_maxpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC, OD, OH, OW,
    PD, PH, PW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    POOL_D: tl.constexpr, POOL_H: tl.constexpr, POOL_W: tl.constexpr,
    inv_div,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    pw = pid_sp % PW
    ph = (pid_sp // PW) % PH
    pd = pid_sp // (PW * PH)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    max_val = tl.full((BLOCK_OC,), -float('inf'), dtype=tl.float32)

    # iterate over pooling window
    for pdi in tl.static_range(0, POOL_D):
        for phi in tl.static_range(0, POOL_H):
            for pwi in tl.static_range(0, POOL_W):
                od = pd * POOL_D + pdi
                oh = ph * POOL_H + phi
                ow = pw * POOL_W + pwi

                acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

                # convolution: sum over IC, KD, KH, KW
                for ic in range(0, IC):
                    for kd in tl.static_range(0, KD):
                        for kh in tl.static_range(0, KH):
                            for kw in tl.static_range(0, KW):
                                id_ = od + kd
                                ih_ = oh + kh
                                iw_ = ow + kw
                                # x[n, ic, id_, ih_, iw_]
                                x_off = ((pid_n * IC + ic) * D + id_) * H * W + ih_ * W + iw_
                                xv = tl.load(x_ptr + x_off)
                                # w[oc, ic, kd, kh, kw]
                                w_off = ((oc_offs * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                                wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                acc += xv * wv

                acc = (acc + bias) * inv_div
                max_val = tl.maximum(max_val, acc)

    # store output [N, OC, PD, PH, PW]
    out_off = ((pid_n * OC + oc_offs) * PD + pd) * PH * PW + ph * PW + pw
    tl.store(out_ptr + out_off, max_val, mask=oc_mask)


@triton.jit
def reduce_avg_bias_sum_kernel(
    pooled_ptr, bias_ptr, out_ptr,
    N, OC, SP,
    BLOCK_SP: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # load bias [OC]
    bias = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)

    # for each oc, sum over SP
    sp_offs = tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < SP

    # pooled shape [N, OC, SP] -> we want sum over SP per (n, oc)
    # then divide by SP -> avg, then add bias, then sum over OC
    # base = pid_n * OC * SP
    ptrs = pooled_ptr + pid_n * OC * SP + oc_offs[:, None] * SP + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    vals = tl.load(ptrs, mask=mask, other=0.0)
    sums = tl.sum(vals, axis=1)  # [BLOCK_OC]
    avg = sums / SP
    avg_plus_bias = avg + bias
    avg_plus_bias = tl.where(oc_mask, avg_plus_bias, 0.0)
    total = tl.sum(avg_plus_bias, axis=0)
    tl.store(out_ptr + pid_n, total)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, D, H, W = x.shape
        KD, KH, KW = self.kernel_size
        OC = self.out_channels
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1
        POOL_D, POOL_H, POOL_W = self.pool_size
        PD = OD // POOL_D
        PH = OH // POOL_H
        PW = OW // POOL_W

        weight = self.conv.weight.contiguous()
        conv_bias = self.conv.bias.contiguous()
        inv_div = 1.0 / self.divisor

        pooled = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=torch.float32)

        BLOCK_OC = 16
        grid = (N, triton.cdiv(OC, BLOCK_OC), PD * PH * PW)

        fused_conv_div_maxpool_kernel[grid](
            x, weight, conv_bias, pooled,
            N, IC, D, H, W,
            OC, OD, OH, OW,
            PD, PH, PW,
            KD, KH, KW,
            POOL_D, POOL_H, POOL_W,
            inv_div,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
        )

        # reduction stage
        SP = PD * PH * PW
        out = torch.empty((N,), device=x.device, dtype=torch.float32)

        # pick block sizes (powers of 2)
        def next_pow2(x):
            p = 1
            while p < x:
                p *= 2
            return p

        BLOCK_SP = next_pow2(SP)
        BLOCK_OC2 = next_pow2(OC)

        bias_flat = self.bias.reshape(-1).contiguous()
        reduce_avg_bias_sum_kernel[(N,)](
            pooled, bias_flat, out,
            N, OC, SP,
            BLOCK_SP=BLOCK_SP,
            BLOCK_OC=BLOCK_OC2,
            num_warps=4,
        )

        return out.view(N, 1, 1, 1)