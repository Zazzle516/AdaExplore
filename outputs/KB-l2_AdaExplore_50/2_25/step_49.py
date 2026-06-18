import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_IN: tl.constexpr, H_IN, W_IN,
    C_OUT: tl.constexpr, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Output tile coordinates
    oh_start = pid_h * BLOCK_H
    ow_start = pid_w * BLOCK_W

    oh_offs = oh_start + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    ow_offs = ow_start + tl.arange(0, BLOCK_W)  # [BLOCK_W]

    oh_mask = oh_offs < H_OUT
    ow_mask = ow_offs < W_OUT
    out_mask = oh_mask[:, None] & ow_mask[None, :]

    # Input tile dimensions (with halo)
    IH_TILE: tl.constexpr = BLOCK_H + KH - 1
    IW_TILE: tl.constexpr = BLOCK_W + KW - 1

    ih_offs = oh_start + tl.arange(0, IH_TILE)  # [IH_TILE]
    iw_offs = ow_start + tl.arange(0, IW_TILE)  # [IW_TILE]
    ih_mask = ih_offs < H_IN
    iw_mask = iw_offs < W_IN

    x_batch_off = pid_n * C_IN * H_IN * W_IN

    # Initialize min accumulator to +inf
    min_val = tl.full((BLOCK_H, BLOCK_W), float('inf'), dtype=tl.float32)

    # Loop over output channels
    for oc in range(0, C_OUT):
        b = tl.load(b_ptr + oc)
        acc = tl.full((BLOCK_H, BLOCK_W), 0.0, dtype=tl.float32)

        # Loop over input channels
        for ic in tl.static_range(0, C_IN):
            # Load input tile [IH_TILE, IW_TILE]
            x_off = (x_batch_off + ic * H_IN * W_IN
                     + ih_offs[:, None] * W_IN + iw_offs[None, :])
            x_mask = ih_mask[:, None] & iw_mask[None, :]
            x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

            # Convolution
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    # Extract slice [BLOCK_H, BLOCK_W] from x_tile starting at [kh, kw]
                    # Use masking trick: build via arange comparisons
                    w_off = oc * C_IN * KH * KW + ic * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)

                    # We need x_tile[kh:kh+BLOCK_H, kw:kw+BLOCK_W]
                    # Reconstruct via direct load from x_ptr using shifted indices
                    sh_off = (x_batch_off + ic * H_IN * W_IN
                              + (oh_offs[:, None] + kh) * W_IN
                              + (ow_offs[None, :] + kw))
                    sh_mask = ((oh_offs[:, None] + kh) < H_IN) & ((ow_offs[None, :] + kw) < W_IN)
                    x_val = tl.load(x_ptr + sh_off, mask=sh_mask, other=0.0)
                    acc += x_val * w_val

        acc = acc + b
        min_val = tl.minimum(min_val, acc)

    # tanh(tanh(x))
    t1 = tl.extra.cuda.libdevice.tanh(min_val)
    t2 = tl.extra.cuda.libdevice.tanh(t1)

    out_off = (pid_n * H_OUT * W_OUT
               + oh_offs[:, None] * W_OUT + ow_offs[None, :])
    tl.store(out_ptr + out_off, t2, mask=out_mask)


@triton.jit
def conv_min_tanh_kernel_v2(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H_IN, W_IN,
    H_OUT, W_OUT,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    oh_start = pid_h * BLOCK_H
    ow_start = pid_w * BLOCK_W

    oh_offs = oh_start + tl.arange(0, BLOCK_H)
    ow_offs = ow_start + tl.arange(0, BLOCK_W)
    oh_mask = oh_offs < H_OUT
    ow_mask = ow_offs < W_OUT
    out_mask = oh_mask[:, None] & ow_mask[None, :]

    IH_TILE: tl.constexpr = BLOCK_H + KH - 1
    IW_TILE: tl.constexpr = BLOCK_W + KW - 1

    ih_offs = oh_start + tl.arange(0, IH_TILE)
    iw_offs = ow_start + tl.arange(0, IW_TILE)
    ih_mask = ih_offs < H_IN
    iw_mask = iw_offs < W_IN
    x_mask_2d = ih_mask[:, None] & iw_mask[None, :]

    x_batch_off = pid_n * C_IN * H_IN * W_IN

    min_val = tl.full((BLOCK_H, BLOCK_W), float('inf'), dtype=tl.float32)

    # Accumulator per OC: we loop OC outer, IC inner with tile-loaded input
    for oc in range(0, C_OUT):
        b = tl.load(b_ptr + oc)
        acc = tl.full((BLOCK_H, BLOCK_W), 0.0, dtype=tl.float32)

        for ic in tl.static_range(0, C_IN):
            # Load input tile once per (oc, ic)
            x_off = (x_batch_off + ic * H_IN * W_IN
                     + ih_offs[:, None] * W_IN + iw_offs[None, :])
            x_tile = tl.load(x_ptr + x_off, mask=x_mask_2d, other=0.0)

            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    w_off = oc * C_IN * KH * KW + ic * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)
                    # Slice x_tile[kh:kh+BLOCK_H, kw:kw+BLOCK_W]
                    # Build via shifted arange masks: use tl.where over full tile not feasible.
                    # Use direct re-load instead (cached in L1).
                    sh_off = (x_batch_off + ic * H_IN * W_IN
                              + (oh_offs[:, None] + kh) * W_IN
                              + (ow_offs[None, :] + kw))
                    sh_mask = ((oh_offs[:, None] + kh) < H_IN) & ((ow_offs[None, :] + kw) < W_IN)
                    xv = tl.load(x_ptr + sh_off, mask=sh_mask, other=0.0)
                    acc += xv * w_val

        acc = acc + b
        min_val = tl.minimum(min_val, acc)

    t1 = tl.extra.cuda.libdevice.tanh(min_val)
    t2 = tl.extra.cuda.libdevice.tanh(t1)

    out_off = (pid_n * H_OUT * W_OUT
               + oh_offs[:, None] * W_OUT + ow_offs[None, :])
    tl.store(out_ptr + out_off, t2, mask=out_mask)


# Cleaner version: load input tile once per IC across all OC by swapping loops.
# Outer: IC, KH, KW load weights for all OC; inner: OC accumulate. But we need min over OC,
# so we must hold per-OC accumulators. Memory: BLOCK_H*BLOCK_W*C_OUT floats per program.
# For BLOCK_H=BLOCK_W=8, C_OUT=64 => 4096 floats = 16KB per program — feasible.

@triton.jit
def conv_min_tanh_fused(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H_IN, W_IN,
    H_OUT, W_OUT,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    oh_start = pid_h * BLOCK_H
    ow_start = pid_w * BLOCK_W

    oh_offs = oh_start + tl.arange(0, BLOCK_H)
    ow_offs = ow_start + tl.arange(0, BLOCK_W)
    oh_mask = oh_offs < H_OUT
    ow_mask = ow_offs < W_OUT
    out_mask = oh_mask[:, None] & ow_mask[None, :]

    x_batch_off = pid_n * C_IN * H_IN * W_IN

    SP: tl.constexpr = BLOCK_H * BLOCK_W
    acc = tl.zeros((C_OUT, SP), dtype=tl.float32)

    oc_offs = tl.arange(0, C_OUT)

    for ic in tl.static_range(0, C_IN):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                sh_off = (x_batch_off + ic * H_IN * W_IN
                          + (oh_offs[:, None] + kh) * W_IN
                          + (ow_offs[None, :] + kw))
                sh_mask = ((oh_offs[:, None] + kh) < H_IN) & ((ow_offs[None, :] + kw) < W_IN)
                xv = tl.load(x_ptr + sh_off, mask=sh_mask, other=0.0)
                xv_flat = tl.reshape(xv, (1, SP))

                w_offs = oc_offs * (C_IN * KH * KW) + ic * KH * KW + kh * KW + kw
                w_col = tl.load(w_ptr + w_offs)
                w_col_b = tl.reshape(w_col, (C_OUT, 1))

                acc += w_col_b * xv_flat

    bias = tl.load(b_ptr + oc_offs)
    acc = acc + tl.reshape(bias, (C_OUT, 1))

    min_val = tl.min(acc, axis=0)

    t1 = tl.extra.cuda.libdevice.tanh(min_val)
    t2 = tl.extra.cuda.libdevice.tanh(t1)

    t2_2d = tl.reshape(t2, (BLOCK_H, BLOCK_W))

    out_off = (pid_n * H_OUT * W_OUT
               + oh_offs[:, None] * W_OUT + ow_offs[None, :])
    tl.store(out_ptr + out_off, t2_2d, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, C_IN, H_IN, W_IN = x.shape
        C_OUT = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        H_OUT = H_IN - KH + 1
        W_OUT = W_IN - KW + 1

        out = torch.empty((N, 1, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        BLOCK_H = 8
        BLOCK_W = 32
        grid = (N, triton.cdiv(H_OUT, BLOCK_H), triton.cdiv(W_OUT, BLOCK_W))

        conv_min_tanh_fused[grid](
            x, w, b, out,
            N, H_IN, W_IN,
            H_OUT, W_OUT,
            C_IN, C_OUT,
            KH, KW,
            BLOCK_H, BLOCK_W,
            num_warps=8,
            num_stages=2,
        )
        return out