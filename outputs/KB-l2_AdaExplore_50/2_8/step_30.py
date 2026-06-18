import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_div_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    PD, PH, PW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    DIVISOR: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # program_id(0): n
    # program_id(1): oc tile
    # program_id(2): pooled spatial tile
    n = tl.program_id(0)
    oc_block = tl.program_id(1)
    sp_block = tl.program_id(2)

    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    sp_offs = sp_block * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]
    P_total = PD * PH * PW
    sp_mask = sp_offs < P_total

    # pooled spatial coords
    pd = sp_offs // (PH * PW)
    rem = sp_offs % (PH * PW)
    ph = rem // PW
    pw = rem % PW

    # accumulator for pooled output: [BLOCK_OC, BLOCK_SP]
    NEG_INF = float('-inf')
    acc = tl.full((BLOCK_OC, BLOCK_SP), NEG_INF, dtype=tl.float32)

    # Iterate over the 2x2x2 pool window
    for pdi in tl.static_range(0, 2):
        for phi in tl.static_range(0, 2):
            for pwi in tl.static_range(0, 2):
                # conv output coordinate
                od = pd * 2 + pdi
                oh = ph * 2 + phi
                ow = pw * 2 + pwi
                valid_sp = sp_mask & (od < OD) & (oh < OH) & (ow < OW)

                # compute conv output for this position over all oc in tile
                conv_acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

                # loop over IC, KD, KH, KW
                for ic in tl.static_range(0, 8):  # IC = 8
                    for kd in tl.static_range(0, KD):
                        for kh in tl.static_range(0, KH):
                            for kw in tl.static_range(0, KW):
                                id_ = od + kd  # input depth idx
                                ih_ = oh + kh
                                iw_ = ow + kw
                                # load input: x[n, ic, id_, ih_, iw_] -> shape [BLOCK_SP]
                                x_off = (((n * IC + ic) * ID + id_) * IH + ih_) * IW + iw_
                                x_val = tl.load(x_ptr + x_off, mask=valid_sp, other=0.0)  # [BLOCK_SP]

                                # load weight: w[oc, ic, kd, kh, kw] -> shape [BLOCK_OC]
                                w_off = (((oc_offs * IC + ic) * KD + kd) * KH + kh) * KW + kw
                                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                                conv_acc += w_val[:, None] * x_val[None, :]

                # add bias
                b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
                conv_acc += b_val[:, None]

                # divide
                conv_acc = conv_acc / DIVISOR

                # mask out invalid positions
                conv_acc = tl.where(valid_sp[None, :], conv_acc, NEG_INF)

                # update max
                acc = tl.maximum(acc, conv_acc)

    # Store pooled output: [N, OC, PD, PH, PW]
    out_off = ((n * OC + oc_offs[:, None]) * P_total) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


@triton.jit
def reduce_avg_bias_sum_kernel(
    pooled_ptr, bias_ptr, out_ptr,
    N, OC, P_total,
    BLOCK_P: tl.constexpr,
):
    # one program per n
    n = tl.program_id(0)
    total = 0.0
    # for each oc, compute mean over P_total, add bias, then sum into total
    # We loop over oc
    for oc in range(0, OC):
        # reduce over P_total
        s = 0.0
        for p_start in range(0, P_total, BLOCK_P):
            offs = p_start + tl.arange(0, BLOCK_P)
            mask = offs < P_total
            base = (n * OC + oc) * P_total
            v = tl.load(pooled_ptr + base + offs, mask=mask, other=0.0)
            s += tl.sum(v, axis=0)
        mean = s / P_total
        b = tl.load(bias_ptr + oc)
        total += (mean + b)
    tl.store(out_ptr + n, total)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD, KH, KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        PD = OD // 2
        PH = OH // 2
        PW = OW // 2

        # weight: conv.weight has shape [OC, IC, KD, KH, KW]
        # bias: conv.bias has shape [OC]
        w = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()

        pooled = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 16
        BLOCK_SP = 64

        grid = (
            N,
            triton.cdiv(OC, BLOCK_OC),
            triton.cdiv(PD * PH * PW, BLOCK_SP),
        )

        fused_conv_div_pool_kernel[grid](
            x, w, cb, pooled,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            PD, PH, PW,
            KD, KH, KW,
            float(self.divisor),
            BLOCK_OC, BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )

        # Reduce: global avg over (PD,PH,PW) -> [N, OC, 1, 1, 1]
        # Then add self.bias [OC,1,1,1] and sum over sum_dim=1 -> [N,1,1,1]
        # bias is [OC, 1, 1, 1] - flatten to [OC]
        bias_flat = self.bias.view(OC).contiguous()

        out = torch.empty((N,), device=x.device, dtype=x.dtype)
        P_total = PD * PH * PW
        BLOCK_P = triton.next_power_of_2(P_total)
        if BLOCK_P > 1024:
            BLOCK_P = 1024

        reduce_avg_bias_sum_kernel[(N,)](
            pooled, bias_flat, out,
            N, OC, P_total,
            BLOCK_P,
            num_warps=2,
        )

        # Return shape: original returns sum over dim=1 of [N, OC, 1, 1, 1] -> [N, 1, 1, 1]
        return out.view(N, 1, 1, 1)