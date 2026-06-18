import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_maxpool_kernel(
    x_ptr,          # input: (N, IC, ID, IH, IW)
    w_ptr,          # weight: (OC, IC, KD, KH, KW)
    b_ptr,          # bias: (OC,)
    out_ptr,        # output: (N, OC, Do, Ho, Wo)
    N, IC, ID, IH, IW,
    OD, OH, OW,
    Do, Ho, Wo,
    OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    TILE_W: tl.constexpr,  # output width tile (4 spatial positions in W direction = 1 pooled output)
):
    # Program layout:
    #   pid_x = pooled (do, ho, wo) linear index
    #   pid_n = batch n
    # For each pooled output cell (covers 4x4x4 conv output), we compute conv for all 64 positions,
    # softmax across OC channels, then max-reduce across the 64 positions per channel.
    pid = tl.program_id(0)
    n = tl.program_id(1)

    total = Do * Ho * Wo
    if pid >= total:
        return

    wo = pid % Wo
    ho = (pid // Wo) % Ho
    do = pid // (Wo * Ho)

    d0 = do * 4
    h0 = ho * 4
    w0 = wo * 4

    oc_offs = tl.arange(0, OC)
    bias = tl.load(b_ptr + oc_offs)

    max_vals = tl.full([OC], -float('inf'), dtype=tl.float32)

    # Process the 4x4x4 window. To exploit input reuse, process one (dk, hk) at a time
    # and load 4 width positions together (plus 2 halo). Conv output positions need
    # input positions [w0..w0+3+KW-1].
    # But for simplicity and to maximize compile-time unrolling, do per-position computation
    # but precompute input loads to local registers.

    for k in tl.static_range(0, 64):
        dk = k // 16
        hk = (k // 4) % 4
        wk = k % 4
        od = d0 + dk
        oh = h0 + hk
        ow = w0 + wk

        acc = bias

        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw
                    for ic in tl.static_range(0, IC_C):
                        x_idx = ((n * IC + ic) * ID + id_) * IH * IW + ih_ * IW + iw_
                        x_val = tl.load(x_ptr + x_idx)
                        w_idx = ((oc_offs * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                        w_val = tl.load(w_ptr + w_idx)
                        acc = acc + x_val * w_val

        m = tl.max(acc, axis=0)
        e = tl.exp(acc - m)
        s = tl.sum(e, axis=0)
        sm = e / s
        max_vals = tl.maximum(max_vals, sm)

    out_base = ((n * OC + 0) * Do + do) * Ho * Wo + ho * Wo + wo
    out_ptrs = out_ptr + out_base + oc_offs * (Do * Ho * Wo)
    tl.store(out_ptrs, max_vals)


@triton.jit
def fused_conv_kernel_tile(
    x_ptr,          # (N, IC, ID, IH, IW)
    w_ptr,          # (OC, IC, KD, KH, KW)
    b_ptr,          # (OC,)
    conv_ptr,       # (N, OC, OD, OH, OW)
    N, IC, ID, IH, IW,
    OD, OH, OW,
    OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    TW: tl.constexpr,  # tile size in W direction
):
    # one program per (n, od, oh, w-tile)
    pid = tl.program_id(0)
    n = tl.program_id(1)

    wtiles = (OW + TW - 1) // TW
    total = OD * OH * wtiles
    if pid >= total:
        return

    wt = pid % wtiles
    oh = (pid // wtiles) % OH
    od = pid // (wtiles * OH)

    w_offs = wt * TW + tl.arange(0, TW)  # (TW,)
    w_mask = w_offs < OW

    oc_offs = tl.arange(0, OC)  # (OC,)
    bias = tl.load(b_ptr + oc_offs)  # (OC,)

    # acc: (OC, TW)
    acc = bias[:, None] + tl.zeros([OC, TW], dtype=tl.float32)

    for kd in tl.static_range(0, KD):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                id_ = od + kd
                ih_ = oh + kh
                iw_offs = w_offs + kw  # (TW,)
                for ic in tl.static_range(0, IC_C):
                    x_base = ((n * IC + ic) * ID + id_) * IH * IW + ih_ * IW
                    x_vals = tl.load(x_ptr + x_base + iw_offs, mask=w_mask, other=0.0)  # (TW,)
                    w_idx = ((oc_offs * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_idx)  # (OC,)
                    acc = acc + w_val[:, None] * x_vals[None, :]

    # Store: conv[n, :, od, oh, w_offs]
    # layout: (N, OC, OD, OH, OW)
    out_base = ((n * OC + 0) * OD + od) * OH * OW + oh * OW
    out_ptrs = conv_ptr + out_base + oc_offs[:, None] * (OD * OH * OW) + w_offs[None, :]
    store_mask = w_mask[None, :] & (oc_offs[:, None] < OC)
    tl.store(out_ptrs, acc, mask=store_mask)


@triton.jit
def softmax_maxpool_kernel(
    conv_ptr,    # (N, OC, OD, OH, OW)
    out_ptr,     # (N, OC, Do, Ho, Wo)
    N, OC, OD, OH, OW,
    Do, Ho, Wo,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = tl.program_id(1)

    total = Do * Ho * Wo
    if pid >= total:
        return

    wo = pid % Wo
    ho = (pid // Wo) % Ho
    do = pid // (Wo * Ho)

    d0 = do * 4
    h0 = ho * 4
    w0 = wo * 4

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < OC

    max_vals = tl.full([BLOCK_C], -float('inf'), dtype=tl.float32)

    for k in tl.static_range(0, 64):
        dk = k // 16
        hk = (k // 4) % 4
        wk = k % 4
        d = d0 + dk
        h = h0 + hk
        w = w0 + wk

        base = ((n * OC + 0) * OD + d) * OH * OW + h * OW + w
        ptrs = conv_ptr + base + c_offs * (OD * OH * OW)
        vals = tl.load(ptrs, mask=c_mask, other=-float('inf'))

        m = tl.max(vals, axis=0)
        e = tl.exp(vals - m)
        e = tl.where(c_mask, e, 0.0)
        s = tl.sum(e, axis=0)
        sm = e / s
        max_vals = tl.maximum(max_vals, sm)

    out_base = ((n * OC + 0) * Do + do) * Ho * Wo + ho * Wo + wo
    out_ptrs = out_ptr + out_base + c_offs * (Do * Ho * Wo)
    tl.store(out_ptrs, max_vals, mask=c_mask)


def fused_pipeline(x, weight, bias, pool_total=4):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape

    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    # Crop conv output so spatial dims are divisible by pool_total
    OD_c = (OD // pool_total) * pool_total
    OH_c = (OH // pool_total) * pool_total
    OW_c = (OW // pool_total) * pool_total

    Do = OD_c // pool_total
    Ho = OH_c // pool_total
    Wo = OW_c // pool_total

    # Allocate conv output buffer (full size); we only read the cropped region in stage 2
    conv = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    # Stage 1: conv
    TW = 16 if OW >= 16 else (8 if OW >= 8 else 4)
    wtiles = (OW + TW - 1) // TW
    grid1 = (OD * OH * wtiles, N)
    fused_conv_kernel_tile[grid1](
        x, weight, bias, conv,
        N, IC, ID, IH, IW,
        OD, OH, OW,
        OC=OC, KD=KD, KH=KH, KW=KW, IC_C=IC, TW=TW,
        num_warps=4, num_stages=2,
    )

    # Stage 2: softmax + max-pool (pool_total^3 window) using cropped region
    out = torch.empty((N, OC, Do, Ho, Wo), device=x.device, dtype=torch.float32)
    BLOCK_C = 1
    while BLOCK_C < OC:
        BLOCK_C *= 2
    BLOCK_C = max(BLOCK_C, 16)

    grid2 = (Do * Ho * Wo, N)
    softmax_maxpool_kernel[grid2](
        conv, out,
        N, OC, OD, OH, OW,
        Do, Ho, Wo,
        BLOCK_C=BLOCK_C,
        num_warps=2, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.pool_total = pool_kernel_size * pool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()
        return fused_pipeline(x, weight, bias, pool_total=self.pool_total)