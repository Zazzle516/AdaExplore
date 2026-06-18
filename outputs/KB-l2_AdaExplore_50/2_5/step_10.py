import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ConvTranspose2d with stride=2, padding=1, kernel=4, output_padding=1
# Input:  (N, IC, H, W)   H=W=256
# Output: (N, OC, OH, OW) OH=OW=512
# 
# For each output (n, oc, oh, ow):
#   sum over (ic, kh, kw) where:
#     ih = (oh + pad - kh) / stride, must be integer in [0, H)
#     iw = (ow + pad - kw) / stride, must be integer in [0, W)
#   acc += input[n, ic, ih, iw] * weight[ic, oc, kh, kw]
# 
# With stride=2, pad=1, ksize=4: for each output pixel, exactly 4 valid (kh,kw) taps
# (2 valid kh, 2 valid kw), unless near boundary.
#
# Strategy: tile by (N*OH_tile, OC) -> GEMM-like
# M = N * OH * OW (output spatial), N_DIM = OC, K = IC * 4 (per-output K is 4 taps * IC)
# But the K depends on output position. Use parity-based grouping.

# We'll launch one program per (n, oh_block, ow_block, oc_block).
# For each output pixel in the tile, the parity (oh+pad)%2 and (ow+pad)%2 selects
# which kh/kw indices are valid.
# Since stride=2, pad=1, kernel=4:
#   valid kh values: kh in {0,1,2,3} with (oh+1-kh) % 2 == 0  -> 2 values
#   ih = (oh+1-kh)//2, need 0 <= ih < H

# Precompute: For oh, the parity p = (oh+1) % 2.
#   If p==0: valid kh in {1, 3}, ih = (oh+1-kh)//2  -> kh=1: ih=oh/2, kh=3: ih=(oh-2)/2
#   If p==1: valid kh in {0, 2}, kh=0: ih=(oh+1)/2, kh=2: ih=(oh-1)/2

# Let's use the simpler "gather" formulation with tile over output spatial.


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=3),
    ],
    key=['IC', 'OC'],
)
@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,        # (N, IC, H, W)
    w_ptr,        # (IC, OC, KH, KW) - PyTorch ConvTranspose2d weight layout
    convbias_ptr, # (OC,) conv's own bias (could be zero)
    bias_ptr,     # (OC,) the subtracted bias
    out_ptr,      # (N, OC, OH, OW)
    N, IC, OC, H, W, OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wic, stride_woc, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,  # tile of output spatial
    BLOCK_N: tl.constexpr,  # tile of OC
):
    # Constants for this conv
    KH: tl.constexpr = 4
    KW: tl.constexpr = 4
    STRIDE: tl.constexpr = 2
    PAD: tl.constexpr = 1

    pid_n = tl.program_id(0)         # batch
    pid_m = tl.program_id(1)         # output spatial tile
    pid_oc = tl.program_id(2)        # OC tile

    # Output spatial offsets within this batch
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # flat output spatial idx
    offs_oc = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)

    OHW = OH * OW
    m_mask = offs_m < OHW
    oc_mask = offs_oc < OC

    oh = offs_m // OW   # [BLOCK_M]
    ow = offs_m % OW

    # For each output pixel, valid kh: kh such that (oh + PAD - kh) % STRIDE == 0
    # and 0 <= ih < H where ih = (oh + PAD - kh) // STRIDE
    # parity of (oh + PAD): if 0, kh must be even-parity matching; specifically
    # (oh + 1 - kh) % 2 == 0  =>  kh % 2 == (oh + 1) % 2
    p_h = (oh + PAD) % STRIDE  # parity, [BLOCK_M]
    p_w = (ow + PAD) % STRIDE

    # kh values: if p_h == 0, kh in {0, 2}? Let's recheck:
    # (oh + 1 - kh) % 2 == 0 -> kh % 2 == (oh+1) % 2
    # If (oh+1)%2 == 0 (oh odd): kh in {0, 2}
    # If (oh+1)%2 == 1 (oh even): kh in {1, 3}
    # So valid kh's parity matches p_h.
    # Two valid kh: kh0 = p_h, kh1 = p_h + 2  (since kh in [0,4))
    # Wait: p_h = (oh+1)%2. If oh odd, p_h=0, kh in {0,2}=>kh0=0=p_h, kh1=2=p_h+2. Good.
    # If oh even, p_h=1, kh in {1,3}=>kh0=1=p_h, kh1=3=p_h+2. Good.

    # ih for each kh
    # ih = (oh + PAD - kh) // STRIDE
    ih0 = (oh + PAD - p_h) // STRIDE       # for kh = p_h
    ih1 = (oh + PAD - (p_h + 2)) // STRIDE # for kh = p_h+2

    iw0 = (ow + PAD - p_w) // STRIDE
    iw1 = (ow + PAD - (p_w + 2)) // STRIDE

    valid_h0 = (ih0 >= 0) & (ih0 < H)
    valid_h1 = (ih1 >= 0) & (ih1 < H)
    valid_w0 = (iw0 >= 0) & (iw0 < W)
    valid_w1 = (iw1 >= 0) & (iw1 < W)

    kh0 = p_h
    kh1 = p_h + 2
    kw0 = p_w
    kw1 = p_w + 2

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For each of 4 (ih, iw, kh, kw) combos, do a small GEMM along IC
    # We loop IC in chunks
    BLOCK_K: tl.constexpr = 32

    # Precompute base x pointer for this batch
    x_batch_ptr = x_ptr + pid_n * stride_xn

    # Pointers to weight: w[ic, oc, kh, kw]
    # We will load w_tile of shape (BLOCK_K, BLOCK_N) for given (kh, kw)
    # For each combo, do K-loop over IC

    # Helper: do one tap (ih, iw, kh, kw) with mask `valid` per-row
    # Inline 4 times to allow constant kh, kw computations.

    # --- Tap 0: (ih0, iw0, kh0, kw0)  valid = valid_h0 & valid_w0 ---
    valid0 = valid_h0 & valid_w0
    # We always compute, but mask loads with valid0 per row
    for ic_start in range(0, IC, BLOCK_K):
        offs_k = ic_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < IC

        # Load x[n, ic, ih0, iw0] -> shape (BLOCK_M, BLOCK_K)
        x_offs = (offs_k[None, :] * stride_xc
                  + ih0[:, None] * stride_xh
                  + iw0[:, None] * stride_xw)
        x_mask = m_mask[:, None] & k_mask[None, :] & valid0[:, None]
        x_vals = tl.load(x_batch_ptr + x_offs, mask=x_mask, other=0.0)

        # Load w[ic, oc, kh0, kw0] -> shape (BLOCK_K, BLOCK_N)
        # Note: kh0, kw0 vary per row of M, so we can't batch by row easily.
        # But across the BLOCK_M, p_h and p_w can be varied. However in our tiling
        # we can choose BLOCK_M tiles aligned so that all rows share same (p_h, p_w).
        # Actually they don't unless we tile carefully. So weight load depends on row.
        # That's expensive (per-row weight load).
        #
        # Better: tile M as (oh_tile, ow_tile) where ow_tile spans contiguous ow.
        # Within an ow run of length BLOCK_W, p_w alternates. Hmm.
        # 
        # Simplification: we can split into 4 sub-tiles based on (p_h, p_w) parity.
        # But for now, let's load weight with row-dependent kh, kw.
        # Each row picks weight[:, :, kh0[row], kw0[row]]. This is a gather.
        w_offs = (offs_k[:, None] * stride_wic
                  + offs_oc[None, :] * stride_woc)  # base for ic, oc
        # Add per-row kh0, kw0 contribution -> need (BLOCK_M, BLOCK_K, BLOCK_N), too big
        # 
        # Alternative: iterate rows? Too slow.
        # 
        # We'll use the fact that p_h and p_w are deterministic from (oh, ow).
        # Split tile so all rows in a tile share parity. We do this by mapping
        # the program's BLOCK_M tile to a contiguous block in (oh, ow), and we
        # accept that parity varies across rows. We work around by computing
        # 4 separate weight loads (one per parity combo) and using mask selects.
        pass
    # The above approach is getting complex; let's restart with a cleaner design.
    # Bail out: we'll use a different kernel below.
    # (This block intentionally does nothing meaningful; the real kernel is _kernel2.)
    pass


# Cleaner design: launch one program per (n, oc_tile, oh, ow_tile).
# For a given oh, p_h is constant across the tile. We iterate ow inside the tile;
# but p_w alternates by ow parity. We split the tile into even-ow and odd-ow lanes.
#
# Even simpler: launch one program per (n, oc_tile, oh, ow). Output is huge though.
#
# Best approach: since p_h depends only on oh and p_w only on ow, we can structure
# the kernel so each program handles a fixed parity pair. Launch grid:
#   (N, OC_tile, OH_tile * 2 (parity h), OW_tile * 2 (parity w))
# This keeps weight loads constant per program.

# Let's code this cleaner version:

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 16, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 32, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'H', 'W'],
)
@triton.jit
def conv_transpose_kernel_v2(
    x_ptr,        # (N, IC, H, W)
    w_ptr,        # (IC, OC, KH, KW)
    bias_ptr,     # (OC,) the subtracted bias (already includes -conv_bias if any)
    out_ptr,      # (N, OC, OH, OW)
    N, IC, OC, H, W, OH, OW,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    KH: tl.constexpr = 4
    KW: tl.constexpr = 4
    STRIDE: tl.constexpr = 2
    PAD: tl.constexpr = 1

    pid_noc = tl.program_id(0)  # n * num_oc_tiles + oc_tile
    pid_oh = tl.program_id(1)   # oh tile index (each tile has 2 oh's: even, odd parity)
    pid_ow = tl.program_id(2)   # ow tile

    num_oc_tiles = tl.cdiv(OC, BLOCK_OC)
    n = pid_noc // num_oc_tiles
    oc_tile = pid_noc % num_oc_tiles

    # Output coords for this tile
    oh_base = pid_oh * BLOCK_OH
    ow_base = pid_ow * BLOCK_OW
    oc_base = oc_tile * BLOCK_OC

    offs_oh = oh_base + tl.arange(0, BLOCK_OH)  # [BLOCK_OH]
    offs_ow = ow_base + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]
    offs_oc = oc_base + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]

    mask_oh = offs_oh < OH
    mask_ow = offs_ow < OW
    mask_oc = offs_oc < OC

    # parity per row/col
    p_h = (offs_oh + PAD) % STRIDE   # [BLOCK_OH]
    p_w = (offs_ow + PAD) % STRIDE   # [BLOCK_OW]

    # ih for each kh in {p_h, p_h+2}
    # ih = (oh + PAD - kh) / STRIDE
    ih_a = (offs_oh + PAD - p_h) // STRIDE          # kh = p_h
    ih_b = (offs_oh + PAD - (p_h + 2)) // STRIDE    # kh = p_h+2
    iw_a = (offs_ow + PAD - p_w) // STRIDE
    iw_b = (offs_ow + PAD - (p_w + 2)) // STRIDE

    valid_ih_a = (ih_a >= 0) & (ih_a < H)
    valid_ih_b = (ih_b >= 0) & (ih_b < H)
    valid_iw_a = (iw_a >= 0) & (iw_a < W)
    valid_iw_b = (iw_b >= 0) & (iw_b < W)

    # We need to gather weight w[:, :, kh, kw] where kh, kw vary per row/col.
    # kh in {p_h, p_h+2} - 2 values per row. kw in {p_w, p_w+2}.
    # Total 4 combos. For each combo, we have (BLOCK_OH, BLOCK_OW) positions but kh
    # depends only on row, kw only on col.

    # For a fixed (kh_choice, kw_choice) in {(a,a),(a,b),(b,a),(b,b)}:
    #  kh value is per-row vector p_h or p_h+2; kw value is per-col vector.
    # Weight tensor w[ic, oc, kh, kw] -> when kh varies per row, we need a 3D gather.
    # However: kh takes at most 2 distinct values across all rows (depending on row parity).
    # If we further tile so all rows in tile share same parity (p_h same), then kh is a
    # single scalar. Similarly for cols.
    #
    # Trick: launch BLOCK_OH=2 with parity-aligned base, but that's tiny.
    # Alternative: split weight load by parity inside kernel using `tl.where`.
    # 
    # We'll do: for each of 4 combos, load weight for both possible kh values and
    # both possible kw values, then select via where. That's 4 weight loads per combo
    # which is wasteful. Instead, let's just load 4 weight slices total (kh in {0,1,2,3} but
    # we only need 2 specific ones) but per row.
    #
    # Cleaner: directly compute kh as a per-row scalar and use tl.load with broadcast.
    # weight offset = ic * (OC*KH*KW) + oc * (KH*KW) + kh * KW + kw
    # If kh varies per row, we can compute the offset as 3D and load (BLOCK_IC, BLOCK_OH, BLOCK_OC)?
    # That's a (K, M, N) tensor — Triton supports 3D, but tl.dot needs 2D.
    #
    # Better: split into 4 separate output computations, one per (kh_a/b, kw_a/b).
    # For kh_a (==p_h), in a tile, p_h is a per-row vector. We load w[:, :, kh_row, kw_col].
    # We can precompute weight_stride and gather: but tl.dot needs 2D inputs.
    # 
    # Alternative final approach: do not use tl.dot. Use elementwise mul and sum.
    # For each (kh, kw) combo:
    #   x_tile: (BLOCK_IC, BLOCK_OH, BLOCK_OW)  [per-tap input]
    #   w_tile: (BLOCK_IC, BLOCK_OH, BLOCK_OW, BLOCK_OC) — too big.
    # 
    # Yet alternative: collapse so kh is constant across the program. Launch with
    # 2 programs per oh tile (one per parity). With BLOCK_OH=4 and parity step=2,
    # rows in tile = oh_base + i*2 + parity. Then p_h = parity (scalar).
    # Same for ow.

    # Pivot to that design in v3:
    pass


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 16, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 32, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 16, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 16, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 16, 'BLOCK_OC': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 32, 'BLOCK_OC': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'H', 'W'],
)
@triton.jit
def conv_transpose_kernel_v3(
    x_ptr,        # (N, IC, H, W)
    w_ptr,        # (IC, OC, KH, KW)
    bias_ptr,     # (OC,) bias to subtract
    out_ptr,      # (N, OC, OH, OW)
    N, IC, OC, H, W, OH, OW,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    KH: tl.constexpr = 4
    KW: tl.constexpr = 4
    STRIDE: tl.constexpr = 2
    PAD: tl.constexpr = 1

    # Grid layout:
    # axis 0: n * num_oc_tiles + oc_tile
    # axis 1: oh_tile_idx * 2 + parity_h    (oh stride is 2*BLOCK_OH; parity selects offset)
    # axis 2: ow_tile_idx * 2 + parity_w
    pid_noc = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    num_oc_tiles = tl.cdiv(OC, BLOCK_OC)
    n = pid_noc // num_oc_tiles
    oc_tile = pid_noc % num_oc_tiles

    parity_h = pid_oh % 2
    oh_tile = pid_oh // 2
    parity_w = pid_ow % 2
    ow_tile = pid_ow // 2

    # Output rows: oh_tile*BLOCK_OH*2 + parity_h + 2*[0..BLOCK_OH)
    # Wait: we want rows where (oh + PAD) % 2 == parity_h  =>  oh % 2 == (parity_h + PAD) % 2
    # With PAD=1: oh % 2 == (parity_h + 1) % 2  -> parity_h=0 -> oh odd; parity_h=1 -> oh even.
    # Easier: just enumerate oh = oh_tile*(2*BLOCK_OH) + parity_h_offset + 2*i
    # Where parity_h_offset is chosen so that (oh + PAD) % 2 == parity_h.
    # If parity_h=0: need oh odd => offset=1
    # If parity_h=1: need oh even => offset=0
    oh_offset = 1 - parity_h
    ow_offset = 1 - parity_w

    offs_oh = oh_tile * (2 * BLOCK_OH) + oh_offset + tl.arange(0, BLOCK_OH) * 2
    offs_ow = ow_tile * (2 * BLOCK_OW) + ow_offset + tl.arange(0, BLOCK_OW) * 2
    offs_oc = oc_tile * BLOCK_OC + tl.arange(0, BLOCK_OC)

    mask_oh = offs_oh < OH
    mask_ow = offs_ow < OW
    mask_oc = offs_oc < OC

    # Now p_h and p_w are constant scalars: parity_h, parity_w
    # kh values: kha = parity_h, khb = parity_h + 2  (constants)
    # kw values: kwa = parity_w, kwb = parity_w + 2
    # ih_a = (oh + 1 - parity_h) // 2; ih_b = (oh - 1 - parity_h) // 2
    # iw_a = (ow + 1 - parity_w) // 2; iw_b = (ow - 1 - parity_w) // 2
    ih_a = (offs_oh + PAD - parity_h) // STRIDE
    ih_b = (offs_oh + PAD - (parity_h + 2)) // STRIDE
    iw_a = (offs_ow + PAD - parity_w) // STRIDE
    iw_b = (offs_ow + PAD - (parity_w + 2)) // STRIDE

    valid_ih_a = (ih_a >= 0) & (ih_a < H) & mask_oh
    valid_ih_b = (ih_b >= 0) & (ih_b < H) & mask_oh
    valid_iw_a = (iw_a >= 0) & (iw_a < W) & mask_ow
    valid_iw_b = (iw_b >= 0) & (iw_b < W) & mask_ow

    # Accumulator: (BLOCK_OH, BLOCK_OW, BLOCK_OC). We need a 3D reduction over IC.
    # tl.dot works on 2D. Reshape: treat (BLOCK_OH * BLOCK_OW) as M dim, BLOCK_OC as N, BLOCK_IC as K.
    # 
    # For each of 4 (kh, kw) combos, do GEMM:
    #   X[m, k] = input[n, k, ih_row(m), iw_col(m)]  where m = oh_idx * BLOCK_OW + ow_idx
    #   W[k, n] = weight[k, n, kh, kw]
    #   acc += X @ W

    M: tl.constexpr = BLOCK_OH * BLOCK_OW
    offs_m = tl.arange(0, M)
    m_oh = offs_m // BLOCK_OW   # which row in tile
    m_ow = offs_m % BLOCK_OW    # which col in tile

    # Gather ih, iw, validity per m
    def_acc_init = tl.zeros((M, BLOCK_OC), dtype=tl.float32)
    acc = def_acc_init

    # x base for this batch
    x_n_ptr = x_ptr + n * IC * H * W

    # Stride consts (contiguous NCHW)
    sxc = H * W
    sxh = W
    sxw = 1

    swic = OC * KH * KW
    swoc = KH * KW
    swkh = KW
    swkw = 1

    # Helper: do one tap with constant kh, kw
    # We'll inline 4 times.

    # ---- Tap (kha, kwa) ----
    kh = parity_h
    kw = parity_w
    # For each m: ih = ih_a[m_oh], iw = iw_a[m_ow], valid = valid_ih_a[m_oh] & valid_iw_a[m_ow]
    ih_m = tl.where(m_oh < BLOCK_OH, tl.load_to_register := ih_a, ih_a)  # placeholder - just use gather
    # Use gather: since ih_a is [BLOCK_OH], we need ih_a[m_oh]. Triton allows indexing 1D with another 1D
    # via broadcast: ih_a[m_oh] is achieved by  tl.where but easier: reshape ih_a to (BLOCK_OH, 1) and broadcast
    # then flatten? Simpler: compute ih_a/iw_a as already-scalar-per-position since we have m_oh, m_ow.
    # 
    # Triton supports indexing by treating ih_a as a 1D tensor and "m_oh" as an integer tensor.
    # But Triton may not directly support gather like numpy. Use broadcasting trick:
    # ih_2d = ih_a[:, None]  (BLOCK_OH, 1)
    # then we want ih_2d at row m_oh: do ih_2d.broadcast_to((BLOCK_OH, BLOCK_OW)).reshape(M)
    # But reshape is allowed? Yes, but easier: build 2D acc and 2D x.
    pass


# OK the above approach got tangled. Let me write a clean 2D version where the
# accumulator is (BLOCK_OH * BLOCK_OW, BLOCK_OC) and we use 2D broadcasting properly.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 16, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 32, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 16, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 16, 'BLOCK_OC': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 16, 'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16, 'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64, 'BLOCK_OC': 32, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'H', 'W'],
)
@triton.jit
def conv_transpose_kernel_final(
    x_ptr,        # (N, IC, H, W) contiguous
    w_ptr,        # (IC, OC, KH, KW) contiguous
    bias_ptr,     # (OC,) bias to subtract
    out_ptr,      # (N, OC, OH, OW) contiguous
    N, IC, OC, H, W, OH, OW,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    KH: tl.constexpr = 4
    KW: tl.constexpr = 4
    STRIDE: tl.constexpr = 2
    PAD: tl.constexpr = 1

    pid_noc = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    num_oc_tiles = tl.cdiv(OC, BLOCK_OC)
    n = pid_noc // num_oc_tiles
    oc_tile = pid_noc % num_oc_tiles

    parity_h = pid_oh % 2
    oh_tile = pid_oh // 2
    parity_w = pid_ow % 2
    ow_tile = pid_ow // 2

    # offset so that (oh + PAD) % 2 == parity_h => oh % 2 == (parity_h + 1) % 2
    oh_offset = 1 - parity_h  # PAD=1
    ow_offset = 1 - parity_w

    rng_oh = tl.arange(0, BLOCK_OH)
    rng_ow = tl.arange(0, BLOCK_OW)
    rng_oc = tl.arange(0, BLOCK_OC)

    offs_oh = oh_tile * (2 * BLOCK_OH) + oh_offset + rng_oh * 2
    offs_ow = ow_tile * (2 * BLOCK_OW) + ow_offset + rng_ow * 2
    offs_oc = oc_tile * BLOCK_OC + rng_oc

    mask_oh = offs_oh < OH
    mask_ow = offs_ow < OW
    mask_oc = offs_oc < OC

    # Compute ih, iw for both kh choices and both kw choices
    # kh in {parity_h, parity_h + 2}; for each give ih = (oh + 1 - kh) / 2
    ih_a = (offs_oh + PAD - parity_h) // STRIDE          # [BLOCK_OH]
    ih_b = (offs_oh + PAD - (parity_h + 2)) // STRIDE
    iw_a = (offs_ow + PAD - parity_w) // STRIDE          # [BLOCK_OW]
    iw_b = (offs_ow + PAD - (parity_w + 2)) // STRIDE

    valid_ih_a = (ih_a >= 0) & (ih_a < H) & mask_oh
    valid_ih_b = (ih_b >= 0) & (ih_b < H) & mask_oh
    valid_iw_a = (iw_a >= 0) & (iw_a < W) & mask_ow
    valid_iw_b = (iw_b >= 0) & (iw_b < W) & mask_ow

    # Accumulator: (BLOCK_OH * BLOCK_OW, BLOCK_OC)
    M: tl.constexpr = BLOCK_OH * BLOCK_OW
    acc = tl.zeros((M, BLOCK_OC), dtype=tl.float32)

    # x batch base
    x_n_base = x_ptr + n * IC * H * W

    # stride helpers for w: w[ic, oc, kh, kw]
    # offset = ic*(OC*KH*KW) + oc*(KH*KW) + kh*KW + kw
    swic = OC * KH * KW
    swoc = KH * KW

    # Loop over IC in chunks
    for ic_start in range(0, IC, BLOCK_IC):
        offs_ic = ic_start + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
        mask_ic = offs_ic < IC

        # ---- Load 4 weight slices for the 4 (kh, kw) combos ----
        # Each is (BLOCK_IC, BLOCK_OC)
        w_base = offs_ic[:, None] * swic + offs_oc[None, :] * swoc
        w_mask = mask_ic[:, None] & mask_oc[None, :]

        w_aa = tl.load(w_ptr + w_base + parity_h * KW + parity_w, mask=w_mask, other=0.0)
        w_ab = tl.load(w_ptr + w_base + parity_h * KW + (parity_w + 2), mask=w_mask, other=0.0)
        w_ba = tl.load(w_ptr + w_base + (parity_h + 2) * KW + parity_w, mask=w_mask, other=0.0)
        w_bb = tl.load(w_ptr + w_base + (parity_h + 2) * KW + (parity_w + 2), mask=w_mask, other=0.0)

        # ---- For each tap, build x tile (M, BLOCK_IC) and do mma ----
        # Tap AA: ih_a, iw_a
        # x[ic, ih, iw] indexing: per m, ih = ih_a[m_oh], iw = iw_a[m_ow]
        # Build (BLOCK_OH, BLOCK_OW, BLOCK_IC) then reshape to (M, BLOCK_IC)?
        # Easier: build ih, iw as 2D (BLOCK_OH, BLOCK_OW) then flatten to (M,)
        ih_aa_2d = ih_a[:, None] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32)
        iw_aa_2d = iw_a[None, :] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32)
        valid_aa_2d = valid_ih_a[:, None] & valid_iw_a[None, :]

        ih_ab_2d = ih_a[:, None] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32)
        iw_ab_2d = iw_b[None, :] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32)
        valid_ab_2d = valid_ih_a[:, None] & valid_iw_b[None, :]

        ih_ba_2d = ih_b[:, None] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32)
        iw_ba_2d = iw_a[None, :] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32)
        valid_ba_2d = valid_ih_b[:, None] & valid_iw_a[None, :]

        ih_bb_2d = ih_b[:, None] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32)
        iw_bb_2d = iw_b[None, :] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32)
        valid_bb_2d = valid_ih_b[:, None] & valid_iw_b[None, :]

        # Flatten to (M,)
        ih_aa = tl.reshape(ih_aa_2d, (M,))
        iw_aa = tl.reshape(iw_aa_2d, (M,))
        valid_aa = tl.reshape(valid_aa_2d, (M,))

        ih_ab = tl.reshape(ih_ab_2d, (M,))
        iw_ab = tl.reshape(iw_ab_2d, (M,))
        valid_ab = tl.reshape(valid_ab_2d, (M,))

        ih_ba = tl.reshape(ih_ba_2d, (M,))
        iw_ba = tl.reshape(iw_ba_2d, (M,))
        valid_ba = tl.reshape(valid_ba_2d, (M,))

        ih_bb = tl.reshape(ih_bb_2d, (M,))
        iw_bb = tl.reshape(iw_bb_2d, (M,))
        valid_bb = tl.reshape(valid_bb_2d, (M,))

        # For each tap, compute x_off (M, BLOCK_IC) = ic*H*W + ih*W + iw
        # Tap AA
        x_off_aa = (offs_ic[None, :] * (H * W)
                    + ih_aa[:, None] * W
                    + iw_aa[:, None])
        x_mask_aa = valid_aa[:, None] & mask_ic[None, :]
        x_aa = tl.load(x_n_base + x_off_aa, mask=x_mask_aa, other=0.0)
        acc += tl.dot(x_aa, w_aa, allow_tf32=True)

        # Tap AB
        x_off_ab = (offs_ic[None, :] * (H * W)
                    + ih_ab[:, None] * W
                    + iw_ab[:, None])
        x_mask_ab = valid_ab[:, None] & mask_ic[None, :]
        x_ab = tl.load(x_n_base + x_off_ab, mask=x_mask_ab, other=0.0)
        acc += tl.dot(x_ab, w_ab, allow_tf32=True)

        # Tap BA
        x_off_ba = (offs_ic[None, :] * (H * W)
                    + ih_ba[:, None] * W
                    + iw_ba[:, None])
        x_mask_ba = valid_ba[:, None] & mask_ic[None, :]
        x_ba = tl.load(x_n_base + x_off_ba, mask=x_mask_ba, other=0.0)
        acc += tl.dot(x_ba, w_ba, allow_tf32=True)

        # Tap BB
        x_off_bb = (offs_ic[None, :] * (H * W)
                    + ih_bb[:, None] * W
                    + iw_bb[:, None])
        x_mask_bb = valid_bb[:, None] & mask_ic[None, :]
        x_bb = tl.load(x_n_base + x_off_bb, mask=x_mask_bb, other=0.0)
        acc += tl.dot(x_bb, w_bb, allow_tf32=True)

    # Add conv bias (already folded into bias_ptr as conv_bias - sub_bias? No: we'll
    # pass the combined bias = sub_bias - conv_bias, then output = tanh(acc - combined_bias)
    # since y = (acc + conv_bias) - sub_bias = acc - (sub_bias - conv_bias)
    b = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    out = acc - b[None, :]
    # tanh
    e_pos = tl.exp(out)
    e_neg = tl.exp(-out)
    out = (e_pos - e_neg) / (e_pos + e_neg)

    # Store: output offset = n*OC*OH*OW + oc*OH*OW + oh*OW + ow
    out_n_base = out_ptr + n * OC * OH * OW
    # Build (M, BLOCK_OC) offsets
    oh_2d = offs_oh[:, None] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32)
    ow_2d = offs_ow[None, :] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32)
    oh_flat = tl.reshape(oh_2d, (M,))
    ow_flat = tl.reshape(ow_2d, (M,))
    oh_valid_2d = mask_oh[:, None] & mask_ow[None, :]
    oh_valid_flat = tl.reshape(oh_valid_2d, (M,))

    out_off = (offs_oc[None, :] * (OH * OW)
               + oh_flat[:, None] * OW
               + ow_flat[:, None])
    out_mask = oh_valid_flat[:, None] & mask_oc[None, :]
    tl.store(out_n_base + out_off, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        # Only fast-path if config matches
        if (self.kernel_size == 4 and self.stride == 2 and self.padding == 1
                and self.output_padding == 1 and x.is_cuda and x.dtype == torch.float32):
            x = x.contiguous()
            N, IC, H, W = x.shape
            OC = self.out_channels
            OH = (H - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
            OW = (W - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

            # Combined bias: subtract_bias - conv_bias  (so out = tanh(acc - combined))
            sub_bias = self.bias.view(-1).contiguous()
            if self.conv_transpose.bias is not None:
                combined_bias = (sub_bias - self.conv_transpose.bias).contiguous()
            else:
                combined_bias = sub_bias.contiguous()

            weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KH, KW)
            out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

            def grid(meta):
                num_oc_tiles = triton.cdiv(OC, meta['BLOCK_OC'])
                num_oh_tiles = triton.cdiv(OH, 2 * meta['BLOCK_OH']) * 2
                num_ow_tiles = triton.cdiv(OW, 2 * meta['BLOCK_OW']) * 2
                return (N * num_oc_tiles, num_oh_tiles, num_ow_tiles)

            conv_transpose_kernel_final[grid](
                x, weight, combined_bias, out,
                N, IC, OC, H, W, OH, OW,
            )
            return out
        else:
            # Fallback
            x = self.conv_transpose(x)
            x = x - self.bias
            x = torch.tanh(x)
            return x