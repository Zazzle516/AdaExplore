import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Strategy: Use input-driven conv-transpose (gather-style).
# For each (n, oc_tile), iterate over input pixels (ih, iw, ic) and kernel (kh, kw).
# For each output position, multiply x[n,ic,ih,iw] * w[ic,oc,kh,kw] and accumulate
# into a per-oc sum over output positions (with bounds-check on output validity).
# We use one program per (n, oc_tile, ic_tile, hw_tile) writing partial sums,
# then a small reduction kernel handles bias/multiplier and final mean.
#
# Better: do GEMM-style. M=OC, N=OH*OW, K=IC*KH*KW (im2col on the fly), with
# one program per (N, OC_tile, HW_tile), then reduce HW.

def _get_conv_configs():
    return [
        triton.Config({'BLOCK_IC': 16, 'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_IC': 32, 'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_IC': 32, 'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_IC': 32, 'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_IC': 64, 'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_IC': 16, 'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_IC': 32, 'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
    ]


@triton.autotune(configs=_get_conv_configs(), key=['OC', 'OH', 'OW', 'IC'])
@triton.jit
def conv_transpose_partial_sum_kernel(
    x_ptr, w_ptr, partial_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    NUM_HW_TILES,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    n = tl.program_id(0)
    oc_tile = tl.program_id(1)
    hw_tile = tl.program_id(2)

    oc_offs = oc_tile * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    hw_offsets = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    oh = hw_offsets // OW
    ow = hw_offsets - oh * OW
    valid_hw = hw_offsets < (OH * OW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        h_num = oh + PH - kh
        ih = h_num // SH
        h_valid = (h_num >= 0) & ((h_num - ih * SH) == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            w_num = ow + PW - kw
            iw = w_num // SW
            w_valid = (w_num >= 0) & ((w_num - iw * SW) == 0) & (iw >= 0) & (iw < IW)
            hw_valid = h_valid & w_valid & valid_hw

            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                x_ptrs = x_ptr + n * (IC * IH * IW) + ic_offs[:, None] * (IH * IW) + ih[None, :] * IW + iw[None, :]
                x_mask = ic_mask[:, None] & hw_valid[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                w_ptrs = w_ptr + ic_offs[:, None] * (OC * KH * KW) + oc_offs[None, :] * (KH * KW) + kh * KW + kw
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                acc += tl.dot(tl.trans(w_vals), x_vals)

    acc = tl.where(valid_hw[None, :], acc, 0.0)
    partial = tl.sum(acc, axis=1)  # (BLOCK_OC,)

    out_ptrs = partial_ptr + n * (OC * NUM_HW_TILES) + oc_offs * NUM_HW_TILES + hw_tile
    tl.store(out_ptrs, partial, mask=oc_mask)


@triton.jit
def reduce_mean_kernel(
    partial_ptr, bias_ptr, out_ptr,
    N, OC, NUM_HW_TILES,
    inv_hw, scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid - n * OC

    base = n * (OC * NUM_HW_TILES) + oc * NUM_HW_TILES

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, NUM_HW_TILES, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < NUM_HW_TILES
        vals = tl.load(partial_ptr + base + offs, mask=mask, other=0.0)
        acc += vals

    total = tl.sum(acc, axis=0)
    bias_val = tl.load(bias_ptr + oc)
    result = (total * inv_hw + bias_val) * scale
    tl.store(out_ptr + pid, result)


def conv_transpose2d_mean_triton(x, weight, bias, stride, padding, output_padding, multiplier):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    SH, SW = stride, stride
    PH, PW = padding, padding
    OPH, OPW = output_padding, output_padding

    OH = (IH - 1) * SH - 2 * PH + KH + OPH
    OW = (IW - 1) * SW - 2 * PW + KW + OPW

    # NUM_HW_TILES depends on BLOCK_HW chosen by autotuner; pick conservative max here
    # Compute it lazily via grid lambda
    out = torch.empty((N, OC), device=x.device, dtype=x.dtype)
    inv_hw = 1.0 / (OH * OW)

    # We need a partial buffer sized per chosen BLOCK_HW. Allocate worst case.
    # Use a fixed BLOCK_HW selection by allocating max possible size (smallest BLOCK_HW=64)
    max_num_hw_tiles = triton.cdiv(OH * OW, 64)
    partial = torch.empty((N, OC, max_num_hw_tiles), device=x.device, dtype=torch.float32)

    def grid(meta):
        nonlocal_num = triton.cdiv(OH * OW, meta['BLOCK_HW'])
        return (N, triton.cdiv(OC, meta['BLOCK_OC']), nonlocal_num)

    # Kernel writes into partial with stride NUM_HW_TILES. We need to pass that
    # value to the kernel. Use a pre-tune to figure out BLOCK_HW.
    # Simpler: launch a wrapper kernel that uses runtime NUM_HW_TILES based on BLOCK_HW.
    # We pass a placeholder NUM_HW_TILES = max_num_hw_tiles and ensure that the
    # kernel writes into [oc * NUM_HW_TILES + hw_tile] where NUM_HW_TILES matches.
    # To keep correctness with autotuner, we restrict configs to BLOCK_HW values
    # that produce valid NUM_HW_TILES; we use one fixed NUM_HW_TILES per launch.
    # Easiest: pick BLOCK_HW manually (no autotune over it).

    # Reset to manually chosen BLOCK_HW
    BLOCK_HW = 128
    NUM_HW_TILES = triton.cdiv(OH * OW, BLOCK_HW)
    partial = torch.empty((N, OC, NUM_HW_TILES), device=x.device, dtype=torch.float32)

    grid2 = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']), NUM_HW_TILES)

    conv_transpose_partial_sum_kernel[grid2](
        x, weight, partial,
        N, IC, IH, IW,
        OC, OH, OW,
        NUM_HW_TILES,
        KH, KW,
        SH, SW,
        PH, PW,
        BLOCK_HW=BLOCK_HW,
    )

    BLOCK_R = triton.next_power_of_2(max(NUM_HW_TILES, 16))
    BLOCK_R = min(BLOCK_R, 1024)
    reduce_mean_kernel[(N * OC,)](
        partial, bias, out,
        N, OC, NUM_HW_TILES,
        inv_hw, multiplier,
        BLOCK=BLOCK_R,
        num_warps=4,
    )

    return out.view(N, OC, 1, 1)


# Fused autotune configs for the v3 kernel (BLOCK_IC = IC, single K iteration)
def _get_conv_configs_v3():
    return [
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
    ]


# Weight layout is [KH, KW, IC, OC] (contiguous in OC).
@triton.autotune(configs=_get_conv_configs_v3(), key=['OC', 'OH', 'OW', 'IC'])
@triton.jit
def conv_transpose_fused_mean_kernel(
    x_ptr, w_ptr, acc_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    oc_tile = tl.program_id(1)
    hw_tile = tl.program_id(2)

    oc_offs = oc_tile * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    hw_offsets = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    oh = hw_offsets // OW
    ow = hw_offsets - oh * OW
    valid_hw = hw_offsets < (OH * OW)

    ic_offs = tl.arange(0, BLOCK_IC)
    ic_mask = ic_offs < IC

    # Accumulator over (BLOCK_HW, BLOCK_OC)
    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    x_base = x_ptr + n * (IC * IH * IW)

    for kh in tl.static_range(0, KH):
        h_num = oh + PH - kh
        ih = h_num // SH
        h_valid = (h_num >= 0) & ((h_num - ih * SH) == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            w_num = ow + PW - kw
            iw = w_num // SW
            w_valid = (w_num >= 0) & ((w_num - iw * SW) == 0) & (iw >= 0) & (iw < IW)
            hw_valid = h_valid & w_valid & valid_hw

            # Load x tile: (BLOCK_HW, BLOCK_IC) - one ic-vector per output position
            x_ptrs = x_base + ic_offs[None, :] * (IH * IW) + ih[:, None] * IW + iw[:, None]
            x_mask = hw_valid[:, None] & ic_mask[None, :]
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # Load weight tile: (BLOCK_IC, BLOCK_OC) - contiguous in OC
            w_ptrs = w_ptr + (kh * KW + kw) * (IC * OC) + ic_offs[:, None] * OC + oc_offs[None, :]
            w_mask = ic_mask[:, None] & oc_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

            acc += tl.dot(x_vals, w_vals)

    # Sum over hw within tile to get (BLOCK_OC,)
    acc = tl.where(valid_hw[:, None], acc, 0.0)
    partial = tl.sum(acc, axis=0)

    # Atomic add into (N, OC) accumulator
    out_ptrs = acc_ptr + n * OC + oc_offs
    tl.atomic_add(out_ptrs, partial, mask=oc_mask)


@triton.jit
def finalize_mean_kernel(
    acc_ptr, bias_ptr, out_ptr,
    N, OC,
    inv_hw, scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < (N * OC)
    oc = offs % OC
    val = tl.load(acc_ptr + offs, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptr + oc, mask=mask, other=0.0)
    result = (val * inv_hw + bias_val) * scale
    tl.store(out_ptr + offs, result, mask=mask)


def conv_transpose2d_mean_triton_v2(x, weight_perm, bias, stride, padding, output_padding, multiplier):
    """
    weight_perm: pre-permuted weight tensor with layout [KH, KW, IC, OC] contiguous.
    """
    N, IC, IH, IW = x.shape
    KH, KW, IC_w, OC = weight_perm.shape
    assert IC == IC_w

    SH, SW = stride, stride
    PH, PW = padding, padding
    OPH, OPW = output_padding, output_padding

    OH = (IH - 1) * SH - 2 * PH + KH + OPH
    OW = (IW - 1) * SW - 2 * PW + KW + OPW

    inv_hw = 1.0 / (OH * OW)

    # Power-of-2 BLOCK_IC for tl.dot; must be >= IC. Pad mask handles the rest.
    BLOCK_IC = triton.next_power_of_2(IC)
    if BLOCK_IC < 16:
        BLOCK_IC = 16

    acc = torch.zeros((N, OC), device=x.device, dtype=torch.float32)
    out = torch.empty((N, OC), device=x.device, dtype=x.dtype)

    grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']), triton.cdiv(OH * OW, meta['BLOCK_HW']))

    conv_transpose_fused_mean_kernel[grid](
        x, weight_perm, acc,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        BLOCK_IC=BLOCK_IC,
    )

    BLOCK_F = 256
    grid_f = (triton.cdiv(N * OC, BLOCK_F),)
    finalize_mean_kernel[grid_f](
        acc, bias, out,
        N, OC,
        inv_hw, multiplier,
        BLOCK=BLOCK_F,
        num_warps=4,
    )

    return out.view(N, OC, 1, 1)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self._cached_weight_perm = None
        self._cached_weight_id = None

    def _get_weight_perm(self):
        w = self.conv_transpose.weight
        # Cache permuted weight: original is [IC, OC, KH, KW]; want [KH, KW, IC, OC] contiguous.
        if self._cached_weight_id != id(w) or self._cached_weight_perm is None:
            self._cached_weight_perm = w.permute(2, 3, 0, 1).contiguous()
            self._cached_weight_id = id(w)
        return self._cached_weight_perm

    def forward(self, x):
        x = x.contiguous()
        weight_perm = self._get_weight_perm()
        bias = self.conv_transpose.bias.contiguous()
        out = conv_transpose2d_mean_triton_v2(
            x, weight_perm, bias, self.stride, self.padding, self.output_padding, self.multiplier
        )
        return out