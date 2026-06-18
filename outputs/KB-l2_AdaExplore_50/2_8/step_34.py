import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused kernel: conv3d + div + maxpool(2x2x2) + global_avg_pool + bias + sum over channels
# Output: per-batch scalar accumulated via atomic adds.
#
# Strategy:
#  - One program per (n, oc_tile, pooled-d-tile)
#  - For each pooled output position (pd, ph, pw):
#      - Compute the 2x2x2 maxpool of conv output across [BLOCK_OC] channels.
#      - The 2x2x2 conv windows are adjacent => input span is (KD+1)x(KH+1)x(KW+1).
#      - For KD=KH=KW=3, that is 4x4x4 input elements per (ic, n, base_pos).
#  - Accumulate sum of max values per channel; at the end divide by num pooled positions,
#    add bias, sum across channels (in-program), atomic_add to out[n].

@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    inv_div,
    BLOCK_OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
):
    n = tl.program_id(0)
    oc_block = tl.program_id(1)
    pd = tl.program_id(2)

    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # Accumulator for sum-over-pooled-positions (for the pd slab) per channel
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    for ph in range(0, PH):
        for pw in range(0, PW):
            # 2x2x2 conv output positions starting at (pd*2, ph*2, pw*2)
            # Initialize max for each channel
            max_val = tl.full([BLOCK_OC], -1e38, dtype=tl.float32)

            # Compute 8 conv outputs per channel and take elementwise max
            # We iterate over input channels and kernel positions.
            # For efficiency: maintain 8 separate accumulators
            c000 = tl.zeros([BLOCK_OC], dtype=tl.float32)
            c001 = tl.zeros([BLOCK_OC], dtype=tl.float32)
            c010 = tl.zeros([BLOCK_OC], dtype=tl.float32)
            c011 = tl.zeros([BLOCK_OC], dtype=tl.float32)
            c100 = tl.zeros([BLOCK_OC], dtype=tl.float32)
            c101 = tl.zeros([BLOCK_OC], dtype=tl.float32)
            c110 = tl.zeros([BLOCK_OC], dtype=tl.float32)
            c111 = tl.zeros([BLOCK_OC], dtype=tl.float32)

            base_d = pd * 2
            base_h = ph * 2
            base_w = pw * 2

            for ic in range(0, IC):
                for kd in range(0, KD):
                    for kh in range(0, KH):
                        for kw in range(0, KW):
                            # weight: [OC, IC, KD, KH, KW]
                            w_off = ((oc_offs * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                            w_v = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                            # 8 input loads for the 2x2x2 conv window block
                            # offset (dd, dh, dw) in {0,1}
                            for dd in tl.static_range(0, 2):
                                for dh in tl.static_range(0, 2):
                                    for dw in tl.static_range(0, 2):
                                        id_ = base_d + dd + kd
                                        ih_ = base_h + dh + kh
                                        iw_ = base_w + dw + kw
                                        x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih_ * IW + iw_
                                        x_v = tl.load(x_ptr + x_off)
                                        prod = x_v * w_v
                                        if (dd == 0) and (dh == 0) and (dw == 0):
                                            c000 += prod
                                        if (dd == 0) and (dh == 0) and (dw == 1):
                                            c001 += prod
                                        if (dd == 0) and (dh == 1) and (dw == 0):
                                            c010 += prod
                                        if (dd == 0) and (dh == 1) and (dw == 1):
                                            c011 += prod
                                        if (dd == 1) and (dh == 0) and (dw == 0):
                                            c100 += prod
                                        if (dd == 1) and (dh == 0) and (dw == 1):
                                            c101 += prod
                                        if (dd == 1) and (dh == 1) and (dw == 0):
                                            c110 += prod
                                        if (dd == 1) and (dh == 1) and (dw == 1):
                                            c111 += prod

            # Add bias and divide
            c000 = (c000 + b_vals) * inv_div
            c001 = (c001 + b_vals) * inv_div
            c010 = (c010 + b_vals) * inv_div
            c011 = (c011 + b_vals) * inv_div
            c100 = (c100 + b_vals) * inv_div
            c101 = (c101 + b_vals) * inv_div
            c110 = (c110 + b_vals) * inv_div
            c111 = (c111 + b_vals) * inv_div

            max_val = tl.maximum(c000, c001)
            max_val = tl.maximum(max_val, c010)
            max_val = tl.maximum(max_val, c011)
            max_val = tl.maximum(max_val, c100)
            max_val = tl.maximum(max_val, c101)
            max_val = tl.maximum(max_val, c110)
            max_val = tl.maximum(max_val, c111)

            acc += max_val

    # Global avg pool divisor
    pool_count = PD * PH * PW
    avg = acc / pool_count

    # Add per-channel bias
    bias_vals = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    avg = avg + bias_vals

    # Sum over channels in this tile (sum_dim=1) and atomic-add to out[n]
    avg_masked = tl.where(oc_mask, avg, 0.0)
    partial = tl.sum(avg_masked, axis=0)
    tl.atomic_add(out_ptr + n, partial)


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
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size, pool_size)
        else:
            self.pool_size = tuple(pool_size)

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD, KH, KW = self.kernel_size
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        PD = OD // self.pool_size[0]
        PH = OH // self.pool_size[1]
        PW = OW // self.pool_size[2]

        # Validate pool size 2x2x2 (kernel assumes 2x2x2)
        if self.pool_size != (2, 2, 2):
            # Fallback to torch
            y = self.conv(x)
            y = y / self.divisor
            y = self.max_pool(y)
            y = self.global_avg_pool(y)
            y = y + self.bias
            y = torch.sum(y, dim=self.sum_dim)
            return y

        weight = self.conv.weight.contiguous()
        conv_bias = self.conv.bias.contiguous()
        bias_flat = self.bias.view(-1).contiguous()

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        BLOCK_OC = 16
        # OC is 16, so one block covers all output channels
        num_oc_blocks = (OC + BLOCK_OC - 1) // BLOCK_OC
        grid = (N, num_oc_blocks, PD)

        fused_kernel[grid](
            x, weight, conv_bias, bias_flat, out,
            N, IC,
            ID, IH, IW,
            OC,
            OD, OH, OW,
            PD, PH, PW,
            1.0 / self.divisor,
            BLOCK_OC=BLOCK_OC,
            KD=KD, KH=KH, KW=KW,
            num_warps=4,
            num_stages=2,
        )

        # Output shape after sum over dim=1 of [N, C, 1, 1, 1] is [N, 1, 1, 1]
        return out.view(N, 1, 1, 1)