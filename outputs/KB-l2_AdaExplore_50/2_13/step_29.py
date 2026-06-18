import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_mean_kernel(
    x_ptr,           # (B, IC, D, H, W)
    w_ptr,           # (IC, OC, 3, 3, 3)  -- ConvTranspose3d weight layout
    cb_ptr,          # (OC,)  conv bias
    out_ptr,         # (B, OC, H, W)
    B, IC, D, H, W, OC,
    inv_D: tl.float32,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # program: (b, oc_block, hw_block)
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    HW = H * W

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW
    h = offs_hw // W
    w = offs_hw % W

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # accumulator (BLOCK_OC, BLOCK_HW)
    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # ConvTranspose3d with stride=1, padding=1, kernel=3 is equivalent to
    # Conv3d with weights flipped over kernel dims, padding=1.
    # weight layout: w[ic, oc, kd, kh, kw], shape (IC, OC, 3, 3, 3)
    # Conv equivalent: y[b,oc,d,h,w] = sum_{ic,kd,kh,kw} x[b,ic,d-kd+1,h-kh+1,w-kw+1] * w[ic,oc,2-kd,2-kh,2-kw]
    # We sum over D as well and divide by D.

    # Loop over kh, kw, ic, kd, d
    # Sum over d of x[b,ic,d-kd+1,h-kh+1,w-kw+1] for d in [0,D)
    # = sum over d_in in valid range of x[b,ic,d_in,h_in,w_in]
    # where d_in = d - kd + 1, so d_in ranges over [1-kd, D-kd]
    # intersected with [0, D-1]
    # For kd=0: d_in in [1, D], clipped to [1, D-1] -> D-1 elements
    # For kd=1: d_in in [0, D-1] -> D elements
    # For kd=2: d_in in [-1, D-2], clipped to [0, D-2] -> D-1 elements

    # Precompute: for each kd, sum_d x[b,ic,d_in,...] but we need per-(h,w) so
    # we just loop. To save flops we precompute the depth sum per (ic, h_in, w_in)
    # but that would change the algorithm. Instead, structure loop to reuse:
    # For fixed (b, ic, h_in, w_in), sum_{d_in=0}^{D-1} x[b,ic,d_in,h_in,w_in] = S
    # Then sum over kd with valid d_in range:
    #   kd=0: S - x[b,ic,0,h_in,w_in]
    #   kd=1: S
    #   kd=2: S - x[b,ic,D-1,h_in,w_in]
    # This way we only need depth sum + 2 boundary slices per (ic,h_in,w_in).

    # We'll compute, for each (kh, kw) and ic:
    #   depth_sum[h_in, w_in] = sum_d x[b,ic,d,h_in,w_in]
    #   x_first[h_in, w_in] = x[b,ic,0,h_in,w_in]
    #   x_last [h_in, w_in] = x[b,ic,D-1,h_in,w_in]
    # Then weight_kd_sum[oc, kd] contributions...
    # Actually simpler: for each (kh,kw,ic), compute three quantities and combine
    # with three weight slices (kd=0,1,2) -> sum_{kd} w_flipped[ic,oc,kd,kh',kw'] * (S - delta_kd)

    # NOTE: This is mathematically the full conv (every multiply-add still happens
    # in expanded form via the depth sum), no algebraic shortcut bypassing flops.
    # Wait — this DOES collapse the depth multiplies. Reread safety contract.
    # The contract forbids pre-reducing along axes a downstream linear reduction
    # collapses. The mean is over D. So we cannot precompute depth sum per ic.
    # We must keep full conv flops.

    # Therefore: iterate full conv, accumulate into mean.
    for kh in tl.static_range(0, 3):
        h_in = h + kh - 1  # because for ConvT with pad=1,k=3,s=1: equivalent conv input idx
        # Actually: convT output[h] = sum_{kh} w[oc,ic,kh] * x_padded[h + kh - pad_eff]
        # For ConvTranspose with stride=1, pad=1, k=3:
        # out[h] = sum_{kh=0..2} w[ic,oc,kh] * x[h - kh + 1]  (when in range)
        # so x index = h - kh + 1. Let's use that.
        pass

    # Re-do loop properly
    for kh in tl.static_range(0, 3):
        h_in = h + 1 - kh  # h - kh + 1
        valid_h = (h_in >= 0) & (h_in < H)
        for kw in tl.static_range(0, 3):
            w_in = w + 1 - kw
            valid_w = (w_in >= 0) & (w_in < W)
            valid_hw = valid_h & valid_w & mask_hw
            for kd in tl.static_range(0, 3):
                # d_in range: kd=0 -> [1, D-1] (D-1 vals), kd=1 -> [0,D-1] (D), kd=2 -> [0, D-2] (D-1)
                d_start = tl.where(kd == 0, 1, 0)
                d_end = tl.where(kd == 2, D - 1, D)  # exclusive
                # loop over d_in
                for d_in in range(0, D):
                    valid_d = (d_in >= d_start) & (d_in < d_end)
                    if valid_d:
                        # gather x[b, ic, d_in, h_in, w_in] for all ic
                        # accumulate sum over ic of x * w[ic, oc, kd, kh, kw]
                        # x_offset depends on ic
                        # We'll loop ic inside.
                        for ic in range(0, IC):
                            x_off = ((pid_b * IC + ic) * D + d_in) * H * W + h_in * W + w_in
                            x_val = tl.load(x_ptr + x_off, mask=valid_hw, other=0.0)
                            # weight: w[ic, oc, kd, kh, kw]
                            w_off = ((ic * OC + offs_oc) * 27) + (kd * 9 + kh * 3 + kw)
                            w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)
                            acc += w_val[:, None] * x_val[None, :]

    acc = acc * inv_D

    # add conv bias (per oc), times D / D = 1, but conv bias is added to every d so mean leaves it unchanged
    cb = tl.load(cb_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + cb[:, None]

    # store: out[b, oc, h, w]
    out_off = pid_b * OC * HW + offs_oc[:, None] * HW + offs_hw[None, :]
    out_mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


@triton.jit
def fused_post_cl_kernel(
    x_ptr,         # (N, C) channels-last flattened
    bias_ptr,      # (C,)
    out_ptr,       # (N, C)
    N, C,
    scaling_factor,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_N

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)

    for i in tl.static_range(0, BLOCK_N):
        row = row_start + i
        row_valid = row < N
        ptrs = x_ptr + row * C + offs_c
        x = tl.load(ptrs, mask=mask_c & row_valid, other=-float('inf'))
        x = x + bias

        x_max = tl.max(x, axis=0)
        x_shift = x - x_max
        e = tl.exp(x_shift)
        e = tl.where(mask_c, e, 0.0)
        s = tl.sum(e, axis=0)
        sm = e / s

        two_x = 2.0 * sm
        e2 = tl.exp(two_x)
        th = (e2 - 1.0) / (e2 + 1.0)
        out = th * scaling_factor

        out_ptrs = out_ptr + row * C + offs_c
        tl.store(out_ptrs, out, mask=mask_c & row_valid)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Use torch's optimized convT then mean — fast on cudnn for this shape
        x = self.conv_transpose(x)            # (B, C, D, H, W)
        x = x.mean(dim=2, keepdim=False)      # (B, C, H, W)

        B, C, H, W = x.shape
        # Permute to (B, H, W, C) so C is contiguous -> coalesced loads
        x_cl = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        out_cl = torch.empty_like(x_cl)

        BLOCK_C = 1
        while BLOCK_C < C:
            BLOCK_C *= 2

        bias_flat = self.bias.view(-1).contiguous()

        N = B * H * W
        BLOCK_N = 4  # rows per program
        grid = ((N + BLOCK_N - 1) // BLOCK_N,)
        fused_post_cl_kernel[grid](
            x_cl, bias_flat, out_cl,
            N, C,
            float(self.scaling_factor),
            BLOCK_C=BLOCK_C,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        # Permute back to (B, C, H, W) then unsqueeze depth
        out = out_cl.permute(0, 3, 1, 2).contiguous()
        return out.unsqueeze(2)