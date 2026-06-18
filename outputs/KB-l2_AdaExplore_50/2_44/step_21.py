import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'IC_TILE': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'IC_TILE': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64, 'IC_TILE': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128, 'IC_TILE': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256, 'IC_TILE': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128, 'IC_TILE': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'IC_TILE': 64}, num_warps=8, num_stages=3),
    ],
    key=['IC', 'OC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv_transpose_fused_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    multiplier,
    NUM_HW_TILES,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    IC_TILE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    ic_offs = tl.arange(0, IC_TILE)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_num = oh + PAD_H - kh
            iw_num = ow + PAD_W - kw
            ih_num_c = tl.where(ih_num >= 0, ih_num, 0)
            iw_num_c = tl.where(iw_num >= 0, iw_num, 0)
            ih = ih_num_c // STRIDE_H
            iw = iw_num_c // STRIDE_W
            valid = (ih_num >= 0) & (iw_num >= 0) & \
                    ((ih_num % STRIDE_H) == 0) & ((iw_num % STRIDE_W) == 0) & \
                    (ih < IH) & (iw < IW) & hw_mask

            for ic_start in range(0, IC, IC_TILE):
                ic_idx = ic_start + ic_offs
                ic_mask = ic_idx < IC

                x_idx = pid_n * (IC * IH * IW) + ic_idx[:, None] * (IH * IW) + (ih * IW + iw)[None, :]
                x_mask = ic_mask[:, None] & valid[None, :]
                x_tile = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

                w_idx = ic_idx[:, None] * (OC * KH * KW) + oc_offs[None, :] * (KH * KW) + (kh * KW + kw)
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_tile = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

                acc += tl.dot(tl.trans(w_tile), x_tile)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]
    acc = acc * multiplier

    # Sum across HW within this tile (masked)
    acc_masked = tl.where(hw_mask[None, :], acc, 0.0)
    partial_sum = tl.sum(acc_masked, axis=1)  # [BLOCK_OC]

    # Store partial sum: shape (N, OC, NUM_HW_TILES)
    out_idx = pid_n * (OC * NUM_HW_TILES) + oc_offs * NUM_HW_TILES + pid_hw
    tl.store(partial_ptr + out_idx, partial_sum, mask=oc_mask)


@triton.jit
def reduce_partial_kernel(
    partial_ptr, out_ptr,
    NUM_TILES, HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*OC
    offs = tl.arange(0, BLOCK)
    mask = offs < NUM_TILES
    vals = tl.load(partial_ptr + pid * NUM_TILES + offs, mask=mask, other=0.0)
    s = tl.sum(vals, axis=0)
    tl.store(out_ptr + pid, s / HW)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()
        bias = self.conv_transpose.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OPH = OPW = self.output_padding

        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW
        HW = OH * OW

        # Allocate partial sums tensor
        # We need to know NUM_HW_TILES, but that depends on autotune. Use max possible.
        # Just allocate based on a chosen tile size - use a wrapper that picks.
        # Simpler: do two-stage with fixed allocation big enough.
        # We'll allocate based on smallest BLOCK_HW in configs = 64.
        max_num_tiles = (HW + 64 - 1) // 64
        partial = torch.empty((N, OC, max_num_tiles), device=x.device, dtype=torch.float32)

        def grid(meta):
            num_hw_tiles = triton.cdiv(HW, meta['BLOCK_HW'])
            return (N, triton.cdiv(OC, meta['BLOCK_OC']), num_hw_tiles)

        # We need NUM_HW_TILES inside the kernel. Pass dynamically via meta lookup.
        # Use a custom approach: pass the stride for partial as max_num_tiles, kernel writes
        # into [pid_n, oc, pid_hw], where pid_hw < actual num_hw_tiles.
        conv_transpose_fused_kernel[grid](
            x, weight, bias, partial,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            SH, SW,
            PH, PW,
            float(self.multiplier),
            max_num_tiles,
        )

        # Now reduce partial[N, OC, :actual_num_tiles] -> out[N, OC] = sum / HW
        # But actual_num_tiles varies based on chosen BLOCK_HW. Use max_num_tiles and
        # rely on zero-initialization... actually partial may have garbage in unused tiles.
        # Solution: zero out partial first.
        # To avoid that overhead, we can determine actual_num_tiles from the chosen config
        # via best_config.
        best_cfg = conv_transpose_fused_kernel.best_config
        actual_block_hw = best_cfg.kwargs['BLOCK_HW']
        actual_num_tiles = (HW + actual_block_hw - 1) // actual_block_hw

        # If actual_num_tiles < max_num_tiles, the partial tensor has stride max_num_tiles
        # but only the first actual_num_tiles entries are valid per (n, oc).
        out_mean = torch.empty((N, OC), device=x.device, dtype=x.dtype)

        # Pick BLOCK as next power of 2 >= actual_num_tiles
        block = 1
        while block < actual_num_tiles:
            block *= 2
        block = max(block, 16)

        reduce_partial_kernel[(N * OC,)](
            partial, out_mean,
            max_num_tiles, HW,  # use max_num_tiles as stride; valid count is actual_num_tiles
            BLOCK=block,
        )
        # Wait - the partial tensor has stride max_num_tiles per (n,oc), but valid count
        # is actual_num_tiles. The kernel loads up to BLOCK elements starting at pid*NUM_TILES.
        # If NUM_TILES (passed) is the stride, mask should be offs < actual_num_tiles,
        # not NUM_TILES. Need to fix: pass actual_num_tiles separately.

        return out_mean.view(N, OC, 1, 1)