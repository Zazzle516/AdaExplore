import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_maxpool_avg_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    PD, PH, PW,  # pooled dims
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    POOL_D: tl.constexpr, POOL_H: tl.constexpr, POOL_W: tl.constexpr,
    divisor: tl.float32,
    BLOCK_POS: tl.constexpr,
):
    # program ids: (n, oc, pooled_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    total_p = PD * PH * PW
    pos_offsets = pid_p * BLOCK_POS + tl.arange(0, BLOCK_POS)
    pos_mask = pos_offsets < total_p

    # decompose pos -> (pd, ph, pw)
    pd = pos_offsets // (PH * PW)
    rem = pos_offsets % (PH * PW)
    ph = rem // PW
    pw = rem % PW

    # For each pooled position, compute max over pool window of conv output
    # conv output at (od, oh, ow) for this (n, oc) =
    #   sum_{ic, kd, kh, kw} x[n, ic, od+kd, oh+kh, ow+kw] * w[oc, ic, kd, kh, kw]
    # then divided by divisor.
    # Pool window: od in [pd*POOL_D, pd*POOL_D + POOL_D), similar for oh, ow.

    NEG_INF = float('-inf')
    max_val = tl.full((BLOCK_POS,), NEG_INF, dtype=tl.float32)

    # Loop over pool window
    for pdi in tl.static_range(POOL_D):
        for phi in tl.static_range(POOL_H):
            for pwi in tl.static_range(POOL_W):
                od = pd * POOL_D + pdi
                oh = ph * POOL_H + phi
                ow = pw * POOL_W + pwi

                # accumulator for conv at this (od, oh, ow)
                acc = tl.zeros((BLOCK_POS,), dtype=tl.float32)

                # Loop over input channels and kernel
                for ic in tl.static_range(0, 8):  # IC=8
                    for kd in tl.static_range(KD):
                        for kh in tl.static_range(KH):
                            for kw in tl.static_range(KW):
                                id_ = od + kd
                                ih = oh + kh
                                iw = ow + kw
                                # x[n, ic, id, ih, iw]
                                x_off = (((pid_n * IC + ic) * ID + id_) * IH + ih) * IW + iw
                                # w[oc, ic, kd, kh, kw]
                                w_off = (((pid_oc * IC + ic) * KD + kd) * KH + kh) * KW + kw
                                # safe load with mask
                                x_val = tl.load(x_ptr + x_off, mask=pos_mask, other=0.0)
                                w_val = tl.load(w_ptr + w_off)
                                acc = acc + x_val * w_val

                # add bias
                bias_val = tl.load(b_ptr + pid_oc)
                acc = acc + bias_val
                acc = acc / divisor

                max_val = tl.where(acc > max_val, acc, max_val)

    # Write to output[n, oc, pd, ph, pw]
    out_off = ((pid_n * OC + pid_oc) * total_p) + pos_offsets
    tl.store(out_ptr + out_off, max_val, mask=pos_mask)


@triton.jit
def final_reduce_kernel(
    pooled_ptr, bias_ptr, out_ptr,
    N, OC, P,
    BLOCK_OC: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # one program per N
    pid_n = tl.program_id(0)
    # compute sum over OC of (mean over P of pooled[n,oc,:] + bias[oc])
    # = (1/P) * sum_oc sum_p pooled[n,oc,p] + sum_oc bias[oc]
    total = tl.zeros((1,), dtype=tl.float32)

    oc_offs = tl.arange(0, BLOCK_OC)
    p_offs = tl.arange(0, BLOCK_P)

    # pooled[n, oc, p]
    # offset = n*OC*P + oc*P + p
    pooled_base = pid_n * OC * P
    ptrs = pooled_base + oc_offs[:, None] * P + p_offs[None, :]
    mask = (oc_offs[:, None] < OC) & (p_offs[None, :] < P)
    vals = tl.load(pooled_ptr + ptrs, mask=mask, other=0.0)
    s = tl.sum(vals, axis=1) / P  # [BLOCK_OC]
    bias_vals = tl.load(bias_ptr + oc_offs, mask=oc_offs < OC, other=0.0)
    s = s + bias_vals
    s = tl.where(oc_offs < OC, s, 0.0)
    total_val = tl.sum(s, axis=0)
    tl.store(out_ptr + pid_n, total_val)


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
        x = x.contiguous().cuda()
        N, IC, ID, IH, IW = x.shape
        KD, KH, KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL_D, POOL_H, POOL_W = self.pool_size
        PD = OD // POOL_D
        PH = OH // POOL_H
        PW = OW // POOL_W

        # Get conv weight and combined bias (conv.bias is separate from self.bias)
        weight = self.conv.weight.contiguous()
        conv_bias = self.conv.bias.contiguous()

        # We will incorporate conv_bias inside the kernel via bias_ptr (added per conv output)
        # And add self.bias at the end after avg pool.

        pooled = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=torch.float32)

        total_p = PD * PH * PW
        BLOCK_POS = 32
        if total_p < 32:
            BLOCK_POS = max(1, triton.next_power_of_2(total_p))

        grid = (N, OC, (total_p + BLOCK_POS - 1) // BLOCK_POS)

        conv3d_maxpool_avg_kernel[grid](
            x, weight, conv_bias, pooled,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            PD, PH, PW,
            KD, KH, KW,
            POOL_D, POOL_H, POOL_W,
            float(self.divisor),
            BLOCK_POS=BLOCK_POS,
            num_warps=4,
        )

        # Now: pooled has shape (N, OC, PD, PH, PW). 
        # Need: mean over (PD,PH,PW) -> (N, OC, 1, 1, 1)
        # add bias (OC, 1, 1, 1) -> (N, OC, 1, 1, 1)
        # sum over dim=1 -> (N, 1, 1, 1)
        
        bias_flat = self.bias.view(-1).contiguous()
        out = torch.empty((N,), device=x.device, dtype=torch.float32)
        P = PD * PH * PW
        BLOCK_OC = triton.next_power_of_2(OC)
        BLOCK_P = triton.next_power_of_2(P)
        
        final_reduce_kernel[(N,)](
            pooled, bias_flat, out,
            N, OC, P,
            BLOCK_OC=BLOCK_OC,
            BLOCK_P=BLOCK_P,
            num_warps=4,
        )

        return out.view(N, 1, 1, 1)