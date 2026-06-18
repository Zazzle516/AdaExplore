import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr,        # [N, IC, ID, IH, IW]
    w_ptr,        # [OC, IC, KD, KH, KW]
    cb_ptr,       # conv bias [OC]
    bias_ptr,     # extra bias [OC]
    out_ptr,      # [N]
    N, IC,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    inv_div: tl.constexpr,
    inv_pool: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    n = tl.program_id(0)
    p_block = tl.program_id(1)

    # Pooled positions handled by this program
    p_offs = p_block * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < (PD * PH * PW)

    pd = p_offs // (PH * PW)
    rem = p_offs % (PH * PW)
    ph = rem // PW
    pw = rem % PW

    # For each pooled position, conv output positions are 2*p + [0,1]
    # We need to compute, for each OC, max over 2x2x2 of conv outputs, then sum -> sum over OC

    # Accumulator: sum over pooled positions of (max-pooled conv output) per OC
    # Then we'll add bias, divide by pool count, sum over OC.
    # But to save memory, let's accumulate per-(p, oc) - actually we sum over p in the end.

    # acc[oc] = sum over p of max-pooled value for this (n, oc)
    # but distributed: each program handles a subset of p's, so use atomic at the end
    # Better: keep acc as [BLOCK_P, OC] then reduce.

    # Per-channel sum across pooled positions in this block
    # Shape: [BLOCK_P] - sum over OC of (avg + bias)
    # Since final result is sum over OC of (global_avg_per_oc + bias_per_oc)
    # = sum_oc[ (sum_p max_p) / (PD*PH*PW) + bias_oc ]
    # = (1/(PD*PH*PW)) * sum_p sum_oc max_p_oc + sum_oc bias_oc
    # The second term is constant per n; we'll add it once via atomic from program (0,0) or precompute.

    # Compute sum_p sum_oc max_p_oc for this block of p's
    oc_range = tl.arange(0, OC)

    # max values per (p, oc): shape [BLOCK_P, OC]
    max_vals = tl.full([BLOCK_P, OC], -1e38, dtype=tl.float32)

    # Loop over 2x2x2 maxpool window
    for dd in tl.static_range(0, 2):
        for dh in tl.static_range(0, 2):
            for dw in tl.static_range(0, 2):
                od = pd * 2 + dd  # [BLOCK_P]
                oh = ph * 2 + dh
                ow = pw * 2 + dw

                # Compute conv output for [BLOCK_P, OC] at these positions
                conv_val = tl.zeros([BLOCK_P, OC], dtype=tl.float32)

                for ic in tl.static_range(0, IC):
                    for kd in tl.static_range(0, KD):
                        for kh in tl.static_range(0, KH):
                            for kw in tl.static_range(0, KW):
                                id_ = od + kd  # [BLOCK_P]
                                ih = oh + kh
                                iw = ow + kw
                                x_off = ((n * IC + ic) * ID + id_) * (IH * IW) + ih * IW + iw  # [BLOCK_P]
                                x_v = tl.load(x_ptr + x_off, mask=p_mask, other=0.0)  # [BLOCK_P]
                                w_off = ((oc_range * IC + ic) * KD + kd) * KH * KW + kh * KW + kw  # [OC]
                                w_v = tl.load(w_ptr + w_off)  # [OC]
                                conv_val += x_v[:, None] * w_v[None, :]

                # Add conv bias, divide by divisor
                cb = tl.load(cb_ptr + oc_range)  # [OC]
                conv_val = (conv_val + cb[None, :]) * inv_div
                max_vals = tl.maximum(max_vals, conv_val)

    # Sum over OC for each p, then sum over p
    # final contribution: (1/pool_count) * sum
    # mask out invalid p's
    max_vals = tl.where(p_mask[:, None], max_vals, 0.0)
    sum_oc = tl.sum(max_vals, axis=1)  # [BLOCK_P]
    total = tl.sum(sum_oc, axis=0)  # scalar

    contrib = total * inv_pool
    tl.atomic_add(out_ptr + n, contrib)


@triton.jit
def add_bias_sum_kernel(bias_ptr, out_ptr, N, OC: tl.constexpr):
    n = tl.program_id(0)
    oc_range = tl.arange(0, OC)
    bias_vals = tl.load(bias_ptr + oc_range)
    s = tl.sum(bias_vals, axis=0)
    tl.atomic_add(out_ptr + n, s)


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
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size,) * 3

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

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        w = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()
        bias_flat = self.bias.view(-1).contiguous()

        BLOCK_P = 32
        total_p = PD * PH * PW
        grid_p = (total_p + BLOCK_P - 1) // BLOCK_P

        inv_div = 1.0 / float(self.divisor)
        inv_pool = 1.0 / float(PD * PH * PW)

        fused_kernel[(N, grid_p)](
            x, w, cb, bias_flat, out,
            N, IC,
            ID, IH, IW,
            OD, OH, OW,
            PD, PH, PW,
            OC,
            KD, KH, KW,
            inv_div,
            inv_pool,
            BLOCK_P,
            num_warps=4,
            num_stages=2,
        )

        # Add the bias sum term once per n
        add_bias_sum_kernel[(N,)](bias_flat, out, N, OC)

        # Output shape: original is [N] after sum over channel dim of [N, 1, 1, 1] -> [N, 1, 1]? Let's check
        # x = global_avg_pool -> [N, OC, 1, 1, 1]
        # x + bias [OC,1,1,1] -> [N, OC, 1, 1, 1]
        # sum(dim=1) -> [N, 1, 1, 1]
        return out.view(N, 1, 1, 1)