import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    PD, PH, PW,  # pooled output dims
    inv_divisor,
    inv_pool_vol,
    BLOCK_P: tl.constexpr,  # pool tile size
    IC_C: tl.constexpr,     # IC as constexpr
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
):
    # grid: (N * OC, num_pool_tiles)
    pid_nc = tl.program_id(0)
    pid_p = tl.program_id(1)

    n = pid_nc // OC
    oc = pid_nc % OC

    P_total = PD * PH * PW
    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < P_total

    pd = p_offs // (PH * PW)
    rem = p_offs % (PH * PW)
    ph = rem // PW
    pw = rem % PW

    # Initialize max accumulator
    NEG_INF = float('-inf')
    max_acc = tl.full((BLOCK_P,), NEG_INF, dtype=tl.float32)

    # Load weights for this oc: [IC, KD, KH, KW]
    # weight pointer: w[oc, ic, kd, kh, kw]
    # We'll iterate over the 8 conv points (2x2x2) in the pool window
    # For each conv point (dd, hh, ww) within pool window:
    #   conv_out at (od, oh, ow) where od = pd*2+dd, oh = ph*2+hh, ow = pw*2+ww
    #   conv_out = sum over ic, kd, kh, kw of x[n, ic, od+kd, oh+kh, ow+kw] * w[oc, ic, kd, kh, kw]

    # Loop over 2x2x2 pool offsets
    for dd in tl.static_range(0, 2):
        for hh in tl.static_range(0, 2):
            for ww in tl.static_range(0, 2):
                od = pd * 2 + dd
                oh = ph * 2 + hh
                ow = pw * 2 + ww

                conv_val = tl.zeros((BLOCK_P,), dtype=tl.float32)

                # Iterate over kernel and IC
                for ic in tl.static_range(0, IC_C):
                    for kd in tl.static_range(0, KD):
                        for kh in tl.static_range(0, KH):
                            for kw in tl.static_range(0, KW):
                                id_idx = od + kd
                                ih_idx = oh + kh
                                iw_idx = ow + kw
                                x_off = ((n * IC + ic) * ID + id_idx) * IH * IW + ih_idx * IW + iw_idx
                                w_off = ((oc * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                                xv = tl.load(x_ptr + x_off, mask=p_mask, other=0.0)
                                wv = tl.load(w_ptr + w_off)
                                conv_val += xv * wv

                # add bias
                bv = tl.load(b_ptr + oc)
                conv_val = conv_val + bv
                # divide
                conv_val = conv_val * inv_divisor
                max_acc = tl.maximum(max_acc, conv_val)

    # Mask out invalid positions
    max_acc = tl.where(p_mask, max_acc, 0.0)

    # Sum across pool tile contribution (partial sum for global avg pool)
    partial_sum = tl.sum(max_acc, axis=0)

    # Atomic add into out[n, oc]
    out_off = n * OC + oc
    tl.atomic_add(out_ptr + out_off, partial_sum * inv_pool_vol)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)*3
        self.divisor = divisor
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size,)*3
        self.sum_dim = sum_dim

        # Match nn.Conv3d default initialization
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(conv.weight.data.clone())
        self.conv_bias = nn.Parameter(conv.bias.data.clone())
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD, KH, KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        PD = OD // self.pool_size[0]
        PH = OH // self.pool_size[1]
        PW = OW // self.pool_size[2]

        inv_divisor = 1.0 / float(self.divisor)
        pool_vol = PD * PH * PW
        inv_pool_vol = 1.0 / float(pool_vol)

        # Output for global avg pool: [N, OC]
        out_pooled = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_P = 64
        P_total = PD * PH * PW
        num_p_tiles = (P_total + BLOCK_P - 1) // BLOCK_P
        grid = (N * OC, num_p_tiles)

        fused_conv_pool_kernel[grid](
            x, self.weight, self.conv_bias, out_pooled,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            PD, PH, PW,
            inv_divisor, inv_pool_vol,
            BLOCK_P=BLOCK_P,
            IC_C=IC,
            KD=KD, KH=KH, KW=KW,
            num_warps=4,
        )

        # out_pooled shape: [N, OC], represents global avg pool output [N, OC, 1, 1, 1] flattened
        # Now add bias (shape [OC, 1, 1, 1]) and sum over sum_dim
        # Original: x shape [N, OC, 1, 1, 1] + bias [OC, 1, 1, 1] -> [N, OC, 1, 1, 1]
        # Then sum over sum_dim
        bias_flat = self.bias.view(OC)  # [OC]
        result = out_pooled + bias_flat.unsqueeze(0)  # [N, OC]

        # Reconstruct to [N, OC, 1, 1, 1] then sum
        result_5d = result.view(N, OC, 1, 1, 1)
        out = torch.sum(result_5d, dim=self.sum_dim)
        return out