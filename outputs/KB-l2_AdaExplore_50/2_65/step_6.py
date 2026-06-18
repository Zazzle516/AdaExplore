import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    PH, PW,
    BLOCK_PW: tl.constexpr,
    POOL: tl.constexpr,
    IC_VAL: tl.constexpr,
    KH_VAL: tl.constexpr,
    KW_VAL: tl.constexpr,
):
    # Grid: (N, OC, PH * ceil(PW/BLOCK_PW))
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    num_pw_tiles = (PW + BLOCK_PW - 1) // BLOCK_PW
    ph = pid_p // num_pw_tiles
    pw_tile = pid_p % num_pw_tiles

    pw_offs = pw_tile * BLOCK_PW + tl.arange(0, BLOCK_PW)  # [BLOCK_PW]
    pw_mask = pw_offs < PW

    # For each pool output column we have POOL conv output columns.
    # base_ow[i] = pw_offs[i] * POOL
    # We need conv output at columns base_ow[i] + kw_idx for kw_idx in [0,POOL)
    # And rows base_oh = ph * POOL + kh_idx for kh_idx in [0,POOL)
    base_oh = ph * POOL
    base_ow = pw_offs * POOL  # [BLOCK_PW]

    # Total conv output columns needed: BLOCK_PW * POOL contiguous-ish columns
    # We'll compute conv at width positions: ow_offs[j] = base_ow[i] + kw_idx
    # Since base_ow is contiguous (stride POOL), the union is just contiguous columns
    # from pw_tile*BLOCK_PW*POOL to (pw_tile+1)*BLOCK_PW*POOL - 1
    ow_base = pw_tile * BLOCK_PW * POOL
    NWCOLS: tl.constexpr = BLOCK_PW * POOL
    ow_offs = ow_base + tl.arange(0, NWCOLS)  # [NWCOLS]
    ow_mask = ow_offs < OW

    bias = tl.load(b_ptr + pid_oc)

    # Accumulator for the sum after sigmoid: scalar
    pool_acc = tl.zeros((BLOCK_PW, POOL), dtype=tl.float32)
    # We'll accumulate conv-sum over POOL rows × POOL cols for each pool output.
    # Approach: for each pool row offset kh_idx, compute conv_row[NWCOLS],
    # then accumulate into per-pool-output sum.

    # For efficiency: total sum across all (kh_idx, kw_idx) in pool window equals
    # sum over POOL rows of conv values per column, then groups of POOL cols sum.
    # Accumulate per-(pool_w, col_in_pool) in 2D tile.

    sum_per_pool = tl.zeros((BLOCK_PW,), dtype=tl.float32)

    for kh_idx in tl.static_range(POOL):
        oh = base_oh + kh_idx  # scalar
        # Compute conv_row over NWCOLS columns
        conv_row = tl.zeros((NWCOLS,), dtype=tl.float32)
        for ic in tl.static_range(IC_VAL):
            for kh in tl.static_range(KH_VAL):
                ih = oh + kh
                for kw in tl.static_range(KW_VAL):
                    iw = ow_offs + kw  # [NWCOLS]
                    x_off = ((pid_n * IC_VAL + ic) * IH + ih) * IW + iw
                    x_val = tl.load(x_ptr + x_off, mask=ow_mask, other=0.0)
                    w_off = ((pid_oc * IC_VAL + ic) * KH_VAL + kh) * KW_VAL + kw
                    w_val = tl.load(w_ptr + w_off)
                    conv_row += x_val * w_val
        conv_row += bias
        # Reshape into [BLOCK_PW, POOL] and add to running pool_acc
        conv_row_2d = tl.reshape(conv_row, (BLOCK_PW, POOL))
        pool_acc += conv_row_2d

    # Average pool
    pooled = pool_acc / (POOL * POOL)  # [BLOCK_PW, POOL]
    # Sum over POOL columns (the pool window cols)
    pooled_sum = tl.sum(pooled, axis=1)  # [BLOCK_PW]
    # Apply sigmoid
    sig = 1.0 / (1.0 + tl.exp(-pooled_sum))
    sig = tl.where(pw_mask, sig, 0.0)
    partial = tl.sum(sig, axis=0)

    tl.atomic_add(out_ptr + pid_n, partial)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.pool_kernel_size
        PH = OH // POOL
        PW = OW // POOL

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        BLOCK_PW = 32
        num_pw_tiles = (PW + BLOCK_PW - 1) // BLOCK_PW
        grid = (N, OC, PH * num_pw_tiles)

        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            BLOCK_PW=BLOCK_PW,
            POOL=POOL,
            IC_VAL=IC,
            KH_VAL=KH,
            KW_VAL=KW,
            num_warps=4,
            num_stages=2,
        )

        return out