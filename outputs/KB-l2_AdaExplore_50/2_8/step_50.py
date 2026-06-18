import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_pool_kernel(
    x_ptr, w_ptr, conv_bias_ptr, out_ptr,
    N, IC, D, H, W,
    OC, KD, KH, KW,
    OD, OH, OW,
    PD, PH, PW,  # pooled dims (after maxpool with stride=2, kernel=2)
    inv_divisor,
    inv_P,
    BLOCK_P: tl.constexpr,
    IC_C: tl.constexpr,
    KD_C: tl.constexpr,
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
):
    # grid: (N, OC, ceil(P / BLOCK_P))
    n = tl.program_id(0)
    oc = tl.program_id(1)
    pblk = tl.program_id(2)

    P = PD * PH * PW

    p_offs = pblk * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < P

    # Decompose pooled index into (pd, ph, pw)
    pd = p_offs // (PH * PW)
    rem = p_offs % (PH * PW)
    ph = rem // PW
    pw = rem % PW

    # For each pooled position, we need 2x2x2 conv outputs starting at:
    # od_base = pd*2, oh_base = ph*2, ow_base = pw*2
    od_base = pd * 2
    oh_base = ph * 2
    ow_base = pw * 2

    # Accumulator for max over 2x2x2 window
    max_val = tl.full([BLOCK_P], -float('inf'), dtype=tl.float32)

    # Iterate over the 8 positions in the 2x2x2 maxpool window
    for dz in tl.static_range(0, 2):
        for dy in tl.static_range(0, 2):
            for dx in tl.static_range(0, 2):
                od = od_base + dz
                oh = oh_base + dy
                ow = ow_base + dx

                # Compute conv output at (n, oc, od, oh, ow)
                conv_val = tl.zeros([BLOCK_P], dtype=tl.float32)

                for ic in tl.static_range(0, IC_C):
                    for kd in tl.static_range(0, KD_C):
                        for kh in tl.static_range(0, KH_C):
                            for kw in tl.static_range(0, KW_C):
                                # input position
                                id_ = od + kd
                                ih = oh + kh
                                iw = ow + kw

                                # bounds check (od, oh, ow could go beyond OD,OH,OW)
                                in_bounds = (od < OD) & (oh < OH) & (ow < OW)

                                x_idx = (((n * IC + ic) * D + id_) * H + ih) * W + iw
                                w_idx = (((oc * IC + ic) * KD_C + kd) * KH_C + kh) * KW_C + kw

                                x_val = tl.load(x_ptr + x_idx, mask=in_bounds, other=0.0)
                                w_val = tl.load(w_ptr + w_idx)

                                conv_val += x_val * w_val

                # Add conv bias and divide
                cb = tl.load(conv_bias_ptr + oc)
                conv_val = (conv_val + cb) * inv_divisor

                # Mask out-of-pool positions
                conv_val = tl.where(p_mask, conv_val, -float('inf'))

                max_val = tl.maximum(max_val, conv_val)

    # Now reduce sum over all pooled positions
    max_val = tl.where(p_mask, max_val, 0.0)
    partial_sum = tl.sum(max_val, axis=0)

    # Multiply by inv_P (for global avg pool)
    partial_sum = partial_sum * inv_P

    # Atomic add into out[n, oc]
    out_idx = n * OC + oc
    tl.atomic_add(out_ptr + out_idx, partial_sum)


@triton.jit
def finalize_kernel(
    partial_ptr, bias_ptr, out_ptr,
    N, OC,
    BLOCK_OC: tl.constexpr,
):
    n = tl.program_id(0)
    oc_offs = tl.arange(0, BLOCK_OC)
    mask = oc_offs < OC

    vals = tl.load(partial_ptr + n * OC + oc_offs, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + oc_offs, mask=mask, other=0.0)
    vals = vals + bias
    s = tl.sum(vals, axis=0)
    tl.store(out_ptr + n, s)


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
        N, IC, D, H, W = x.shape
        KD, KH, KW = self.kernel_size
        OC = self.out_channels
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1
        PKD, PKH, PKW = self.pool_size
        PD = OD // PKD
        PH = OH // PKH
        PW = OW // PKW

        P = PD * PH * PW
        inv_P = 1.0 / P
        inv_div = 1.0 / self.divisor

        weight = self.conv.weight.contiguous()
        conv_bias = self.conv.bias.contiguous()

        # partial sum: (N, OC)
        partial = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_P = 128
        grid = (N, OC, (P + BLOCK_P - 1) // BLOCK_P)

        fused_conv_pool_kernel[grid](
            x, weight, conv_bias, partial,
            N, IC, D, H, W,
            OC, KD, KH, KW,
            OD, OH, OW,
            PD, PH, PW,
            inv_div, inv_P,
            BLOCK_P=BLOCK_P,
            IC_C=IC, KD_C=KD, KH_C=KH, KW_C=KW,
            num_warps=4,
        )

        # Finalize: add bias (per-OC) and sum over OC
        bias_flat = self.bias.view(-1).contiguous()
        out = torch.empty((N,), device=x.device, dtype=torch.float32)

        BLOCK_OC = triton.next_power_of_2(OC)
        finalize_kernel[(N,)](
            partial, bias_flat, out,
            N, OC,
            BLOCK_OC=BLOCK_OC,
        )

        # Original output shape: after sum(dim=1) on (N, OC, 1, 1, 1) -> (N, 1, 1, 1)
        return out.view(N, 1, 1, 1)