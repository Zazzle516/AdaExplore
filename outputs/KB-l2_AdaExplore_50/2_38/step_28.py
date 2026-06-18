import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused pool + conv_transpose3d + clamp kernel (gather formulation)
# ---------------------------------------------------------------------------
# After avg_pool3d with kernel=2 (stride=2), the pooled input shape is
#   (B, IC, D_in/2, H_in/2, W_in/2)
# Then conv_transpose3d with kernel=3, stride=2, padding=1, output_padding=1
# produces output shape:
#   D_out = (D_p-1)*2 - 2 + 3 + 1 = 2*D_p
# i.e. same as the original input spatial dims.
#
# For each output voxel (od, oh, ow), the valid input voxels (id, ih, iw) and
# kernel taps satisfy: id*stride - pad + kd = od (similarly for h, w)
# so kd = od + pad - id*stride. For kd in [0, K-1], id = (od + pad - kd)/stride
# requires (od + pad - kd) % stride == 0.
# ---------------------------------------------------------------------------


@triton.jit
def fused_pool_convt_clamp_kernel(
    x_ptr,           # input (B, IC, Di, Hi, Wi)  -- before pool
    w_ptr,           # weight (IC, OC, K, K, K)
    b_ptr,           # bias (OC,)
    out_ptr,         # output (B, OC, Do, Ho, Wo)
    B, IC, OC,
    Di, Hi, Wi,
    Dp, Hp, Wp,      # pooled dims = Di/2, Hi/2, Wi/2
    Do, Ho, Wo,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    POOL: tl.constexpr,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # grid: (B, Do, ceil(Ho*Wo / BLOCK_SP) * ceil(OC/BLOCK_OC))
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_oc_sp = tl.program_id(2)

    n_sp_blocks = tl.cdiv(Ho * Wo, BLOCK_SP)
    pid_oc = pid_oc_sp // n_sp_blocks
    pid_sp = pid_oc_sp % n_sp_blocks

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)        # (BLOCK_OC,)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)        # (BLOCK_SP,)
    oh = sp_offs // Wo
    ow = sp_offs % Wo

    oc_mask = oc_offs < OC                                       # (BLOCK_OC,)
    sp_mask = sp_offs < (Ho * Wo)                                # (BLOCK_SP,)

    # Accumulator
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    od = pid_d  # output depth index

    # Loop over kernel taps. For each (kd, kh, kw) determine the corresponding
    # input voxel index and accumulate.
    for kd in tl.static_range(0, K):
        num_d = od + PAD - kd
        id_q = num_d // STRIDE
        id_r = num_d - id_q * STRIDE
        d_valid = (id_r == 0) & (id_q >= 0) & (id_q < Dp)

        for kh in tl.static_range(0, K):
            num_h = oh + PAD - kh                                # (BLOCK_SP,)
            ih_q = num_h // STRIDE
            ih_r = num_h - ih_q * STRIDE
            h_valid = (ih_r == 0) & (ih_q >= 0) & (ih_q < Hp)

            for kw in tl.static_range(0, K):
                num_w = ow + PAD - kw
                iw_q = num_w // STRIDE
                iw_r = num_w - iw_q * STRIDE
                w_valid = (iw_r == 0) & (iw_q >= 0) & (iw_q < Wp)

                pos_valid = d_valid & h_valid & w_valid          # (BLOCK_SP,)

                # Loop over input channels and accumulate
                # weight[ic, oc, kd, kh, kw]
                # pooled[b, ic, id_q, ih_q, iw_q]
                for ic in range(0, IC):
                    # Load weight slice for this (ic, kd, kh, kw): shape (OC,)
                    w_slot = ((ic * OC + oc_offs) * K * K * K) + (kd * K * K + kh * K + kw)
                    w_val = tl.load(w_ptr + w_slot, mask=oc_mask, other=0.0)  # (BLOCK_OC,)

                    # Load pooled value for this (ic, id_q, ih_q, iw_q): shape (BLOCK_SP,)
                    # pooled value = mean of 2x2x2 block of x at
                    #   (b, ic, id_q*2..id_q*2+1, ih_q*2..+1, iw_q*2..+1)
                    base_x = ((pid_b * IC + ic) * Di) 
                    # Compute pooled value inline
                    id_in = id_q * POOL
                    ih_in = ih_q * POOL                          # (BLOCK_SP,)
                    iw_in = iw_q * POOL

                    pool_acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)
                    inv_pool3 = 1.0 / (POOL * POOL * POOL)
                    for pd in tl.static_range(0, POOL):
                        for ph in tl.static_range(0, POOL):
                            for pw in tl.static_range(0, POOL):
                                x_d = id_in + pd
                                x_h = ih_in + ph
                                x_w = iw_in + pw
                                x_off = (((pid_b * IC + ic) * Di + x_d) * Hi + x_h) * Wi + x_w
                                xv = tl.load(x_ptr + x_off, mask=pos_valid & sp_mask, other=0.0)
                                pool_acc += xv
                    pooled = pool_acc * inv_pool3                # (BLOCK_SP,)

                    # outer-product: (BLOCK_OC,1) * (1,BLOCK_SP) -> (BLOCK_OC,BLOCK_SP)
                    acc += w_val[:, None] * pooled[None, :]

    # Add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]

    # Clamp
    acc = tl.minimum(tl.maximum(acc, clamp_min), clamp_max)

    # Store: out[b, oc, od, oh, ow]
    out_off = ((((pid_b * OC + oc_offs[:, None]) * Do + od) * Ho + oh[None, :]) * Wo + ow[None, :])
    store_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=store_mask)


# ---------------------------------------------------------------------------
# Softmax + scale kernel (per-channel row, online softmax)
# ---------------------------------------------------------------------------
@triton.jit
def softmax_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    B, C, S,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    row_start = (b * C + c) * S
    scale = tl.load(scale_ptr + c)

    max_val = -float('inf')
    sum_exp = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
        cur_max = tl.max(v, axis=0)
        new_max = tl.maximum(max_val, cur_max)
        e = tl.exp(v - new_max)
        e = tl.where(mask, e, 0.0)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(e, axis=0)
        max_val = new_max

    inv = 1.0 / sum_exp

    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        e = tl.exp(v - max_val) * inv * scale
        tl.store(out_ptr + row_start + idx, e, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, pool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # Keep an nn.ConvTranspose3d to own the weights/bias (matches reference init)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding,
                                                 output_padding=output_padding)
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = x.contiguous()
        B, IC, Di, Hi, Wi = x.shape
        OC = self.out_channels
        K = self.kernel_size
        STRIDE = self.stride
        PAD = self.padding
        POOL = self.pool_kernel_size

        Dp = Di // POOL
        Hp = Hi // POOL
        Wp = Wi // POOL

        # output spatial dims of conv_transpose3d
        Do = (Dp - 1) * STRIDE - 2 * PAD + K + self.output_padding
        Ho = (Hp - 1) * STRIDE - 2 * PAD + K + self.output_padding
        Wo = (Wp - 1) * STRIDE - 2 * PAD + K + self.output_padding

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, K, K, K)
        bias = self.conv_transpose.bias.contiguous()      # (OC,)

        # Intermediate buffer for clamped conv_transpose output
        mid = torch.empty((B, OC, Do, Ho, Wo), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 64

        n_sp_blocks = (Ho * Wo + BLOCK_SP - 1) // BLOCK_SP
        n_oc_blocks = (OC + BLOCK_OC - 1) // BLOCK_OC

        grid = (B, Do, n_sp_blocks * n_oc_blocks)

        fused_pool_convt_clamp_kernel[grid](
            x, weight, bias, mid,
            B, IC, OC,
            Di, Hi, Wi,
            Dp, Hp, Wp,
            Do, Ho, Wo,
            K=K, STRIDE=STRIDE, PAD=PAD, POOL=POOL,
            clamp_min=self.clamp_min, clamp_max=self.clamp_max,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            num_warps=4, num_stages=2,
        )

        # Softmax + scale across spatial dims
        S = Do * Ho * Wo
        x_flat = mid.view(B * OC, S)
        out = torch.empty_like(x_flat)
        scale_flat = self.scale.view(-1).contiguous()

        if S >= 4096:
            BLOCK = 4096
            num_warps = 16
        elif S >= 2048:
            BLOCK = 2048
            num_warps = 8
        else:
            BLOCK = 1024
            num_warps = 8

        grid2 = (B * OC,)
        softmax_scale_kernel[grid2](
            x_flat, scale_flat, out,
            B, OC, S,
            BLOCK=BLOCK, num_warps=num_warps, num_stages=2,
        )

        return out.view(B, OC, Do, Ho, Wo)