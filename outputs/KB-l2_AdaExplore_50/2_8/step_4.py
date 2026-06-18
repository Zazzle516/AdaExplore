import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_pool_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC, D, H, W,
    OC, KD, KH, KW,
    OD, OH, OW,         # conv output dims
    PD, PH, PW,         # pooled dims
    inv_div,
    inv_pool_count,     # 1.0 / (PD*PH*PW)
    BLOCK_OC: tl.constexpr,
    POOL_D: tl.constexpr,
    POOL_H: tl.constexpr,
    POOL_W: tl.constexpr,
    IC_KDHW: tl.constexpr,  # IC * KD * KH * KW
):
    # one program per (n, pd, ph, pw) -- computes conv for the pool window across all OC,
    # max-pools, accumulates avg into per-(n) atomic sum (after sum over OC and bias).
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)

    P_total = PD * PH * PW
    pd = pid // (PH * PW)
    rem = pid % (PH * PW)
    ph = rem // PW
    pw = rem % PW

    # base output coords (top-left of pool window in conv-output space)
    od_base = pd * POOL_D
    oh_base = ph * POOL_H
    ow_base = pw * POOL_W

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # accumulator for max over the pool window: shape [BLOCK_OC]
    NEG_INF = -1.0e30
    max_vals = tl.full((BLOCK_OC,), NEG_INF, dtype=tl.float32)

    # Iterate over pool positions
    for pp in tl.static_range(0, POOL_D * POOL_H * POOL_W):
        ld = pp // (POOL_H * POOL_W)
        lr = pp % (POOL_H * POOL_W)
        lh = lr // POOL_W
        lw = lr % POOL_W

        od = od_base + ld
        oh = oh_base + lh
        ow = ow_base + lw

        # compute conv output for all OC at (n, od, oh, ow)
        # conv_val[oc] = sum_{ic,kd,kh,kw} x[n, ic, od+kd, oh+kh, ow+kw] * w[oc, ic, kd, kh, kw] + b[oc]
        acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

        for k in range(0, IC_KDHW):
            ic = k // (KD * KH * KW)
            kr = k % (KD * KH * KW)
            kd = kr // (KH * KW)
            kr2 = kr % (KH * KW)
            kh = kr2 // KW
            kw = kr2 % KW

            in_d = od + kd
            in_h = oh + kh
            in_w = ow + kw

            x_off = ((pid_n * IC + ic) * D + in_d) * H * W + in_h * W + in_w
            x_val = tl.load(x_ptr + x_off)

            # weight: [OC, IC, KD, KH, KW]
            w_off = oc_offs * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
            w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

            acc += x_val * w_val

        # add bias and divide
        b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
        acc = (acc + b_val) * inv_div

        max_vals = tl.maximum(max_vals, acc)

    # avg contribution: max_vals * inv_pool_count, then add bias_ptr[oc] *(scaled by 1/P_total? no)
    # global avg pool over P_total positions: sum(max_vals across pool positions) / P_total
    # Each program contributes max_vals * (1/P_total) to the (n, oc) avg.
    # Then output = sum_oc(avg[n,oc] + bias[oc]) for n.
    # We accumulate into a per-(n, oc) partial sum via atomic_add.

    contrib = max_vals * inv_pool_count  # this position's contribution to avg[n,oc]

    # atomic add to partial[n, oc]
    partial_off = pid_n * OC + oc_offs
    tl.atomic_add(out_ptr + partial_off, contrib, mask=oc_mask)


@triton.jit
def final_reduce_kernel(
    partial_ptr, bias_ptr, out_ptr,
    N, OC,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    p = tl.load(partial_ptr + pid_n * OC + oc_offs, mask=oc_mask, other=0.0)
    b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    val = p + b
    s = tl.sum(val, axis=0)
    tl.store(out_ptr + pid_n, s)


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
        N = x.shape[0]
        IC = self.in_channels
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        OC = self.out_channels
        KD, KH, KW = self.kernel_size
        POOL_D, POOL_H, POOL_W = self.pool_size

        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1
        PD = OD // POOL_D
        PH = OH // POOL_H
        PW = OW // POOL_W

        # If shapes don't fit fast path, fall back to torch
        if PD * POOL_D != OD or PH * POOL_H != OH or PW * POOL_W != OW:
            y = self.conv(x) / self.divisor
            y = self.max_pool(y)
            y = self.global_avg_pool(y)
            y = y + self.bias
            return torch.sum(y, dim=self.sum_dim)

        weight = self.conv.weight.contiguous()
        conv_bias = self.conv.bias.contiguous()
        bias_flat = self.bias.view(-1).contiguous()

        # partial[n, oc]
        partial = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        IC_KDHW = IC * KD * KH * KW
        BLOCK_OC = triton.next_power_of_2(OC)
        if BLOCK_OC < 16:
            BLOCK_OC = 16

        P_total = PD * PH * PW
        inv_div = 1.0 / float(self.divisor)
        inv_pool_count = 1.0 / float(P_total)

        grid = (P_total, N)
        fused_conv_pool_kernel[grid](
            x, weight, conv_bias, bias_flat, partial,
            N, IC, D, H, W,
            OC, KD, KH, KW,
            OD, OH, OW,
            PD, PH, PW,
            inv_div, inv_pool_count,
            BLOCK_OC=BLOCK_OC,
            POOL_D=POOL_D, POOL_H=POOL_H, POOL_W=POOL_W,
            IC_KDHW=IC_KDHW,
            num_warps=4,
        )

        # Final: out[n] = sum_oc(partial[n, oc] + bias_flat[oc])
        out = torch.empty((N,), device=x.device, dtype=torch.float32)
        final_reduce_kernel[(N,)](
            partial, bias_flat, out,
            N, OC,
            BLOCK_OC=BLOCK_OC,
            num_warps=1,
        )

        # Output shape: original sums dim=1 of tensor shape (N, OC, 1, 1, 1) -> (N, 1, 1, 1)
        return out.view(N, 1, 1, 1)