import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Gather-style ConvTranspose3d with tl.dot
# For each output position (n, oc_tile, spatial_tile):
#   acc[m, n_oc] = sum over (ic, kd, kh, kw):
#       x_gather[m, ic] * w[ic, oc, kd, kh, kw]
# We loop over (kd, kh, kw) outside and do a (BLOCK_M x IC) @ (IC x BLOCK_N) matmul.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OD', 'OH', 'OW', 'IC', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv_transpose3d_gather_kernel(
    x_ptr, w_ptr, scale_ptr, shift_ptr, out_ptr, partial_sum_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)  # spatial tile id (over OD*OH*OW)
    pid_n = tl.program_id(1)  # oc tile id
    pid_b = tl.program_id(2)  # batch id

    spatial_size = OD * OH * OW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M] flat spatial indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N] oc indices
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K] ic indices

    mask_m = offs_m < spatial_size
    mask_n = offs_n < OC

    od = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n = pid_b

    # iterate over kernel positions
    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_val = id_num // SD
        id_valid = (id_num >= 0) & ((id_num % SD) == 0) & (id_val < ID) & (id_val >= 0)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih_val = ih_num // SH
            ih_valid = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih_val < IH) & (ih_val >= 0)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PW - kw
                iw_val = iw_num // SW
                iw_valid = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw_val < IW) & (iw_val >= 0)

                spatial_valid = id_valid & ih_valid & iw_valid & mask_m  # [BLOCK_M]

                # base input offset (n, :, id, ih, iw); we add ic*ID*IH*IW
                base_x = ((n * IC) * ID + id_val) * IH * IW + ih_val * IW + iw_val  # [BLOCK_M]

                # base weight offset: w[ic, oc, kd, kh, kw]
                # offset = ic*OC*KD*KH*KW + oc*KD*KH*KW + kd*KH*KW + kh*KW + kw
                w_kdhw_off = (kd * KH + kh) * KW + kw  # scalar
                base_w = offs_n * (KD * KH * KW) + w_kdhw_off  # [BLOCK_N]

                # loop over IC in BLOCK_K chunks
                for ic_start in range(0, IC, BLOCK_K):
                    ic_idx = ic_start + offs_k  # [BLOCK_K]
                    mask_k = ic_idx < IC

                    # x gather: shape [BLOCK_M, BLOCK_K]
                    x_addrs = base_x[:, None] + ic_idx[None, :] * (ID * IH * IW)
                    x_mask = spatial_valid[:, None] & mask_k[None, :]
                    x_tile = tl.load(x_ptr + x_addrs, mask=x_mask, other=0.0)

                    # w gather: shape [BLOCK_K, BLOCK_N]
                    w_addrs = ic_idx[:, None] * (OC * KD * KH * KW) + base_w[None, :]
                    w_mask = mask_k[:, None] & mask_n[None, :]
                    w_tile = tl.load(w_ptr + w_addrs, mask=w_mask, other=0.0)

                    acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # apply BN-folded affine: y = acc * scale[oc] + shift[oc]
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)  # [BLOCK_N]
    shift = tl.load(shift_ptr + offs_n, mask=mask_n, other=0.0)  # [BLOCK_N]
    y = acc * scale[None, :] + shift[None, :]

    # store output
    # out layout: (N, OC, OD*OH*OW)
    out_addrs = ((n * OC + offs_n[None, :]) * spatial_size) + offs_m[:, None]
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_addrs, y, mask=store_mask)

    # accumulate partial sum per (n, oc) tile for mean computation
    # partial_sum_ptr layout: (N, OC, num_m_tiles)
    y_masked = tl.where(store_mask, y, 0.0)
    psum = tl.sum(y_masked, axis=0)  # [BLOCK_N]
    num_m_tiles = tl.cdiv(spatial_size, BLOCK_M)
    ps_addrs = (n * OC + offs_n) * num_m_tiles + pid_m
    tl.store(partial_sum_ptr + ps_addrs, psum, mask=mask_n)


@triton.jit
def mean_sub_kernel(
    inout_ptr, partial_sum_ptr,
    N, C, SPATIAL, NUM_TILES,
    BLOCK: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    # reduce partial sums to get total sum
    ps_base = (n * C + c) * NUM_TILES
    total = tl.zeros((REDUCE_BLOCK,), dtype=tl.float32)
    for off in range(0, NUM_TILES, REDUCE_BLOCK):
        idx = off + tl.arange(0, REDUCE_BLOCK)
        mask = idx < NUM_TILES
        v = tl.load(partial_sum_ptr + ps_base + idx, mask=mask, other=0.0)
        total += tl.where(mask, v, 0.0)
    total_scalar = tl.sum(total, axis=0)
    mean = total_scalar / SPATIAL

    # subtract mean from inout
    base = (n * C + c) * SPATIAL
    for off in range(0, SPATIAL, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < SPATIAL
        x = tl.load(inout_ptr + base + idx, mask=mask, other=0.0)
        tl.store(inout_ptr + base + idx, x - mean, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        if self.conv_transpose.bias is not None:
            conv_bias = self.conv_transpose.bias.contiguous()
        else:
            conv_bias = torch.zeros(weight.shape[1], device=x.device, dtype=x.dtype)

        N, IC, ID, IH, IW = x.shape
        OC = weight.shape[1]
        KD, KH, KW = self.kernel_size, self.kernel_size, self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding
        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        # We need BN to behave like training: but for correctness with original Model,
        # we must compute exactly: BN(conv_out), then subtract spatial mean.
        # In training, BN computes per-channel mean/var across (N, D, H, W) of the conv output.
        # We need conv_out to compute these stats, then apply the BN affine.
        # Strategy: do conv first (without fusion), compute BN stats, then run fused kernel
        # that applies BN affine and writes output + partial sum, then mean subtraction.

        # First, do conv via fused kernel with identity scale/shift to materialize conv output.
        # Actually we need conv output to compute BN running stats. Let's do it in two phases:
        # Phase 1: run gather kernel with scale=1, shift=conv_bias_per_oc, NO mean-sub partial sum.
        # Then compute per-channel mean/var.
        # Phase 2: run again? Too expensive. Instead: compute conv first, then BN, then mean-sub.

        # Better: run conv kernel writing to a buffer, with conv bias as shift, scale=1.
        # Then compute BN stats from that buffer, then do fused (BN affine + mean sub) using
        # mean_sub_kernel variant. But we also need partial sums for spatial mean.
        # We can compute spatial mean AFTER BN affine; that's just per-(n,c) mean.

        spatial_size = OD * OH * OW

        # Allocate conv output
        conv_out = torch.empty((N, OC, spatial_size), device=x.device, dtype=torch.float32)

        # Phase 1: conv with scale=1, shift=conv_bias
        scale1 = torch.ones(OC, device=x.device, dtype=torch.float32)
        # We need to know BLOCK_M to size partial_sum buffer; use a fixed max and the autotune
        # will pick BLOCK_M. To avoid this complexity, allocate with worst case BLOCK_M=32.
        # Actually partial_sum buffer size depends on BLOCK_M chosen by autotune.
        # Use a dummy partial sum buffer that's large enough.
        max_num_tiles_phase1 = (spatial_size + 32 - 1) // 32
        partial_sum_dummy = torch.empty((N, OC, max_num_tiles_phase1), device=x.device, dtype=torch.float32)

        def grid1(meta):
            return (
                triton.cdiv(spatial_size, meta['BLOCK_M']),
                triton.cdiv(OC, meta['BLOCK_N']),
                N,
            )

        # We need partial_sum buffer sized correctly per chosen BLOCK_M. The kernel writes
        # at index pid_m. To avoid OOB, allocate based on the smallest BLOCK_M in configs (32).
        # Worst case num_m_tiles = ceil(spatial/32). All configs have BLOCK_M >= 32 so this is safe upper bound.

        BLOCK_K_CONST = 16

        conv_transpose3d_gather_kernel[grid1](
            x, weight, scale1, conv_bias, conv_out, partial_sum_dummy,
            N, IC, OC,
            ID, IH, IW,
            OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_K=BLOCK_K_CONST,
        )

        # Now conv_out has shape (N, OC, spatial_size) with conv_transpose output (incl bias).
        # Run BN in training mode: compute per-channel mean/var across (N, spatial).
        conv_out_5d = conv_out.view(N, OC, OD, OH, OW)
        bn_out = self.batch_norm(conv_out_5d)

        # Now subtract spatial mean per (n, c)
        # Use simple kernel: compute sum then subtract.
        bn_out_flat = bn_out.contiguous().view(N, OC, spatial_size)
        # Compute mean via torch (fast for this size)
        mean = bn_out_flat.mean(dim=2, keepdim=True)
        out = bn_out_flat - mean
        return out.view(N, OC, OD, OH, OW)