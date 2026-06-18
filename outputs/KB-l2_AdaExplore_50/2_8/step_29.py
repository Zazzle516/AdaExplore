import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_div_maxpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    PD, PH, PW,
    divisor,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
):
    # one program per (N, OC, pool_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_tile = tl.program_id(2)

    tile_offs = pid_tile * BLOCK_TILE + tl.arange(0, BLOCK_TILE)
    pool_total = PD * PH * PW
    mask_tile = tile_offs < pool_total

    pd = tile_offs // (PH * PW)
    rem = tile_offs % (PH * PW)
    ph = rem // PW
    pw = rem % PW

    # pooled position -> conv output region (2x2x2 max pool, stride 2)
    od_base = pd * 2
    oh_base = ph * 2
    ow_base = pw * 2

    # bias
    b = tl.load(b_ptr + pid_oc).to(tl.float32)

    # accumulator for max over 2x2x2 conv outputs
    NEG_INF = -1.0e30
    max_val = tl.full((BLOCK_TILE,), NEG_INF, dtype=tl.float32)

    # Loop over the 2x2x2 maxpool window
    for dz in tl.static_range(0, 2):
        for dy in tl.static_range(0, 2):
            for dx in tl.static_range(0, 2):
                od = od_base + dz
                oh = oh_base + dy
                ow = ow_base + dx
                # conv at (od, oh, ow)
                acc = tl.zeros((BLOCK_TILE,), dtype=tl.float32)
                for ic in tl.static_range(0, IC_C):
                    for kd in tl.static_range(0, KD):
                        for kh in tl.static_range(0, KH):
                            for kw in tl.static_range(0, KW):
                                id_ = od + kd
                                ih = oh + kh
                                iw = ow + kw
                                x_off = (((pid_n * IC + ic) * ID + id_) * IH + ih) * IW + iw
                                w_off = (((pid_oc * IC_C + ic) * KD + kd) * KH + kh) * KW + kw
                                xv = tl.load(x_ptr + x_off, mask=mask_tile, other=0.0).to(tl.float32)
                                wv = tl.load(w_ptr + w_off).to(tl.float32)
                                acc = acc + xv * wv
                acc = acc + b
                acc = acc / divisor
                max_val = tl.maximum(max_val, acc)

    # write out shape [N, OC, PD, PH, PW]
    out_off = ((pid_n * OC + pid_oc) * pool_total) + tile_offs
    tl.store(out_ptr + out_off, max_val, mask=mask_tile)


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
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD, KH, KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        PD = OD // self.pool_size[0]
        PH = OH // self.pool_size[1]
        PW = OW // self.pool_size[2]

        weight = self.conv.weight.contiguous()
        cbias = self.conv.bias.contiguous()

        pool_total = PD * PH * PW
        out = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=torch.float32)

        BLOCK_TILE = 64
        num_tiles = (pool_total + BLOCK_TILE - 1) // BLOCK_TILE
        grid = (N, OC, num_tiles)

        conv_div_maxpool_kernel[grid](
            x, weight, cbias, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            PD, PH, PW,
            float(self.divisor),
            KD, KH, KW,
            IC,
            BLOCK_TILE,
            num_warps=4,
        )

        # global avg pool over (PD, PH, PW)
        gap = out.mean(dim=(2, 3, 4), keepdim=True)  # [N, OC, 1, 1, 1]
        gap = gap + self.bias
        result = torch.sum(gap, dim=self.sum_dim)
        return result