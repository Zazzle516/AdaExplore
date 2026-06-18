import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Output dims after conv (no padding, stride 1): D_out=14, H_out=62, W_out=62
# After maxpool(2): D=7, H=31, W=31. Total = 7*31*31 = 6727 windows
# After global avg pool: scalar per (n, oc). Then +bias, sum over OC -> (N,)
#
# Strategy: one program per n. Loop over all pool windows, for each window
# compute the 2x2x2 max over conv outputs. Sum across all windows and all OC,
# then divide by (divisor * total_pool_windows) and add sum(bias) at the end.
#
# Conv output at (oc, d, h, w) = sum_{ic, kd, kh, kw} x[n,ic,d+kd,h+kh,w+kw] * W[oc,ic,kd,kh,kw] + b[oc]
#
# We'll precompute on host: total_windows = D_p*H_p*W_p, and at the end:
# result[n] = (sum over oc, over windows, of maxpool(conv/divisor)) / total_windows + sum(bias)
#
# So we need sum over windows of max over 2x2x2 of conv outputs, summed over OC.
#
# For each pool window at (pd, ph, pw), the 8 conv positions are
# (2pd+dd, 2ph+dh, 2pw+dw) for dd,dh,dw in {0,1}.
#
# For each (n, pool_window), compute 8 conv values per OC, take max over 8, sum over OC.
#
# Kernel: one program per (n, pool_window_tile). Inside, loop over windows in tile,
# compute conv for all OC and all 8 sub-positions, max-reduce, sum over OC, accumulate.

@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, bias_sum_ptr, out_ptr,
    N, IC: tl.constexpr, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    OC: tl.constexpr, KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    D_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    D_p: tl.constexpr, H_p: tl.constexpr, W_p: tl.constexpr,
    total_windows: tl.constexpr,
    divisor: tl.constexpr,
    inv_total: tl.constexpr,
    BLOCK_WIN: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_w = tl.program_id(1)

    win_offs = pid_w * BLOCK_WIN + tl.arange(0, BLOCK_WIN)
    win_mask = win_offs < total_windows

    # Decompose window index into (pd, ph, pw)
    pd = win_offs // (H_p * W_p)
    rem = win_offs % (H_p * W_p)
    ph = rem // W_p
    pw = rem % W_p

    # Base conv output positions
    d_base = pd * 2  # [BLOCK_WIN]
    h_base = ph * 2
    w_base = pw * 2

    # Accumulator: sum over OC of max over 8 subpositions of conv value
    acc = tl.zeros([BLOCK_WIN], dtype=tl.float32)

    # Loop over OC
    for oc in tl.static_range(0, OC):
        # For each of the 8 subpositions, compute conv value
        # conv[bw, sub] = sum_{ic,kd,kh,kw} x[n, ic, d_base+dd+kd, h_base+dh+kh, w_base+dw+kw] * W[oc,ic,kd,kh,kw]
        # We compute all 8 sub-positions.

        # Initialize 8 accumulators per win position
        c000 = tl.zeros([BLOCK_WIN], dtype=tl.float32)
        c001 = tl.zeros([BLOCK_WIN], dtype=tl.float32)
        c010 = tl.zeros([BLOCK_WIN], dtype=tl.float32)
        c011 = tl.zeros([BLOCK_WIN], dtype=tl.float32)
        c100 = tl.zeros([BLOCK_WIN], dtype=tl.float32)
        c101 = tl.zeros([BLOCK_WIN], dtype=tl.float32)
        c110 = tl.zeros([BLOCK_WIN], dtype=tl.float32)
        c111 = tl.zeros([BLOCK_WIN], dtype=tl.float32)

        for ic in tl.static_range(0, IC):
            for kd in tl.static_range(0, KD):
                for kh in tl.static_range(0, KH):
                    for kw in tl.static_range(0, KW):
                        w_val = tl.load(w_ptr + ((oc * IC + ic) * KD + kd) * KH * KW + kh * KW + kw)
                        # Conv position for sub=(dd,dh,dw): d = d_base+dd+kd, h = h_base+dh+kh, w = w_base+dw+kw
                        # We need to load x at 8 sub-positions. But many overlap: for kd, the d index is d_base+kd (sub dd=0) or d_base+1+kd (sub dd=1). For KD=3, distinct d's = {kd, kd+1} where kd in 0..2 => d offsets 0..3. Similar for h,w. So 4*4*4=64 loads per (oc,ic) — but we can just do 8 loads per sub.
                        # Simpler: do 8 separate loads.
                        base_offset = (pid_n * IC + ic) * D * H * W

                        d0 = d_base + kd
                        d1 = d_base + 1 + kd
                        h0 = h_base + kh
                        h1 = h_base + 1 + kh
                        w0 = w_base + kw
                        w1 = w_base + 1 + kw

                        # Load 8 values
                        x000 = tl.load(x_ptr + base_offset + d0 * H * W + h0 * W + w0, mask=win_mask, other=0.0)
                        x001 = tl.load(x_ptr + base_offset + d0 * H * W + h0 * W + w1, mask=win_mask, other=0.0)
                        x010 = tl.load(x_ptr + base_offset + d0 * H * W + h1 * W + w0, mask=win_mask, other=0.0)
                        x011 = tl.load(x_ptr + base_offset + d0 * H * W + h1 * W + w1, mask=win_mask, other=0.0)
                        x100 = tl.load(x_ptr + base_offset + d1 * H * W + h0 * W + w0, mask=win_mask, other=0.0)
                        x101 = tl.load(x_ptr + base_offset + d1 * H * W + h0 * W + w1, mask=win_mask, other=0.0)
                        x110 = tl.load(x_ptr + base_offset + d1 * H * W + h1 * W + w0, mask=win_mask, other=0.0)
                        x111 = tl.load(x_ptr + base_offset + d1 * H * W + h1 * W + w1, mask=win_mask, other=0.0)

                        c000 += x000 * w_val
                        c001 += x001 * w_val
                        c010 += x010 * w_val
                        c011 += x011 * w_val
                        c100 += x100 * w_val
                        c101 += x101 * w_val
                        c110 += x110 * w_val
                        c111 += x111 * w_val

        # Add bias
        b_val = tl.load(b_ptr + oc)
        c000 = (c000 + b_val) / divisor
        c001 = (c001 + b_val) / divisor
        c010 = (c010 + b_val) / divisor
        c011 = (c011 + b_val) / divisor
        c100 = (c100 + b_val) / divisor
        c101 = (c101 + b_val) / divisor
        c110 = (c110 + b_val) / divisor
        c111 = (c111 + b_val) / divisor

        # Max over 8
        m = tl.maximum(c000, c001)
        m = tl.maximum(m, c010)
        m = tl.maximum(m, c011)
        m = tl.maximum(m, c100)
        m = tl.maximum(m, c101)
        m = tl.maximum(m, c110)
        m = tl.maximum(m, c111)

        acc += m

    # Mask out invalid windows
    acc = tl.where(win_mask, acc, 0.0)

    # Sum over windows in this tile, accumulate to output via atomic add
    partial = tl.sum(acc, axis=0)
    # Multiply by inv_total here for the avg pool
    partial = partial * inv_total

    tl.atomic_add(out_ptr + pid_n, partial)


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
        N, IC, D, H, W = x.shape
        KD, KH, KW = self.kernel_size
        OC = self.out_channels
        D_out = D - KD + 1
        H_out = H - KH + 1
        W_out = W - KW + 1
        pD, pH, pW = self.pool_size
        D_p = D_out // pD
        H_p = H_out // pH
        W_p = W_out // pW
        total_windows = D_p * H_p * W_p

        # bias_sum is sum over OC of self.bias broadcasted; but the bias is (OC,1,1,1)
        # After avg_pool -> (N,OC,1,1,1), + bias -> (N,OC,1,1,1), sum over dim=1 -> (N,1,1,1)
        # So output shape is (N,1,1,1). The constant part = sum(bias) added per N.
        bias_sum = self.bias.sum().item()

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        weight = self.conv.weight.contiguous()
        conv_bias = self.conv.bias.contiguous()

        BLOCK_WIN = 64
        grid = (N, (total_windows + BLOCK_WIN - 1) // BLOCK_WIN)

        fused_kernel[grid](
            x, weight, conv_bias, None, out,
            N, IC, D, H, W,
            OC, KD, KH, KW,
            D_out, H_out, W_out,
            D_p, H_p, W_p,
            total_windows,
            float(self.divisor),
            1.0 / float(total_windows),
            BLOCK_WIN=BLOCK_WIN,
            num_warps=4,
            num_stages=2,
        )

        out = out + bias_sum
        return out.view(N, 1, 1, 1)