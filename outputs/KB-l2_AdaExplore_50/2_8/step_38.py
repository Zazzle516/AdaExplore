import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC: tl.constexpr,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    inv_div: tl.constexpr,
    inv_pool: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # One program per (n, oc, pooled-spatial tile)
    n = tl.program_id(0)
    oc = tl.program_id(1)
    p_block = tl.program_id(2)

    p_offs = p_block * BLOCK_P + tl.arange(0, BLOCK_P)
    P_total = PD * PH * PW
    p_mask = p_offs < P_total

    pd = p_offs // (PH * PW)
    rem = p_offs % (PH * PW)
    ph = rem // PW
    pw = rem % PW

    # base output coords for each pooled position
    od_base = pd * 2
    oh_base = ph * 2
    ow_base = pw * 2

    # Load bias_conv for this oc once
    b_val = tl.load(b_ptr + oc)

    # Accumulator over pooled positions: hold max per position
    max_vals = tl.full([BLOCK_P], -1e38, dtype=tl.float32)

    # Loop over the 2x2x2 maxpool window
    for dd in tl.static_range(0, 2):
        for dh in tl.static_range(0, 2):
            for dw in tl.static_range(0, 2):
                od = od_base + dd
                oh = oh_base + dh
                ow = ow_base + dw

                conv_val = tl.zeros([BLOCK_P], dtype=tl.float32)

                for ic in tl.static_range(0, IC):
                    for kd in tl.static_range(0, KD):
                        for kh in tl.static_range(0, KH):
                            for kw in tl.static_range(0, KW):
                                id_ = od + kd
                                ih = oh + kh
                                iw = ow + kw
                                x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
                                x_v = tl.load(x_ptr + x_off, mask=p_mask, other=0.0)
                                w_off = ((oc * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                                w_v = tl.load(w_ptr + w_off)
                                conv_val += x_v * w_v

                conv_val = (conv_val + b_val) * inv_div
                max_vals = tl.maximum(max_vals, conv_val)

    # Sum over pooled positions (for global avg)
    max_vals = tl.where(p_mask, max_vals, 0.0)
    partial = tl.sum(max_vals, axis=0)

    # avg = partial / P_total ; then + bias_per_oc ; sum into out[n]
    # We do per-channel scaling here and atomic_add to out[n]
    bias_oc = tl.load(bias_ptr + oc)
    # Only one program-axis-2 should add the bias to avoid duplicating it.
    # Add bias only when p_block == 0.
    contrib = partial * inv_pool
    if p_block == 0:
        contrib = contrib + bias_oc

    tl.atomic_add(out_ptr + n, contrib)


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
        ID, IH, IW = x.shape[2], x.shape[3], x.shape[4]
        KD, KH, KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        PD = OD // 2
        PH = OH // 2
        PW = OW // 2

        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        bias = self.bias.contiguous().view(-1)

        out = torch.zeros((N,), device=x.device, dtype=torch.float32)

        BLOCK_P = 64
        P_total = PD * PH * PW
        grid_p = (P_total + BLOCK_P - 1) // BLOCK_P
        grid = (N, OC, grid_p)

        fused_kernel[grid](
            x, w, b, bias, out,
            N, IC,
            ID, IH, IW,
            OC,
            OD, OH, OW,
            PD, PH, PW,
            KD, KH, KW,
            1.0 / self.divisor,
            1.0 / (PD * PH * PW),
            BLOCK_P,
            num_warps=4,
            num_stages=2,
        )

        return out.view(N, 1, 1, 1)