import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC, OD, OH, OW,         # conv output spatial dims
    PD, PH, PW,             # pooled spatial dims (after maxpool)
    divisor,
    BLOCK_OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    POOL_D: tl.constexpr, POOL_H: tl.constexpr, POOL_W: tl.constexpr,
    TILE_P: tl.constexpr,   # pooled positions per program
):
    # one program per (n, pooled-tile)
    pid_n = tl.program_id(0)
    pid_p = tl.program_id(1)

    p_total = PD * PH * PW
    p_offsets = pid_p * TILE_P + tl.arange(0, TILE_P)
    p_mask = p_offsets < p_total

    # decompose pooled position -> (pd, ph, pw)
    pd = p_offsets // (PH * PW)
    rem = p_offsets % (PH * PW)
    ph = rem // PW
    pw = rem % PW

    # conv output start position for this pool window
    od_start = pd * POOL_D
    oh_start = ph * POOL_H
    ow_start = pw * POOL_W

    # accumulator: [TILE_P, BLOCK_OC] for max-pool result
    oc_offsets = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < OC

    # we compute for each pooled position, the max over the POOL_D*POOL_H*POOL_W conv outputs
    # Each conv output requires sum over IC*KD*KH*KW
    NEG_INF = -1.0e30
    maxv = tl.full((TILE_P, BLOCK_OC), NEG_INF, dtype=tl.float32)

    # iterate over pool window
    for pdi in tl.static_range(0, POOL_D):
        for phi in tl.static_range(0, POOL_H):
            for pwi in tl.static_range(0, POOL_W):
                od = od_start + pdi  # [TILE_P]
                oh = oh_start + phi
                ow = ow_start + pwi

                # accumulator for conv: [TILE_P, BLOCK_OC]
                acc = tl.zeros((TILE_P, BLOCK_OC), dtype=tl.float32)

                # iterate over IC, KD, KH, KW
                for ic in tl.static_range(0, 8):  # IC = 8
                    for kd in tl.static_range(0, KD):
                        for kh in tl.static_range(0, KH):
                            for kw in tl.static_range(0, KW):
                                id_ = od + kd  # [TILE_P]
                                ih = oh + kh
                                iw = ow + kw
                                # load input value: [TILE_P]
                                x_idx = (pid_n * IC * D * H * W
                                         + ic * D * H * W
                                         + id_ * H * W
                                         + ih * W
                                         + iw)
                                xv = tl.load(x_ptr + x_idx, mask=p_mask, other=0.0)
                                # load weight: [BLOCK_OC]
                                w_idx = (oc_offsets * (IC * KD * KH * KW)
                                         + ic * KD * KH * KW
                                         + kd * KH * KW
                                         + kh * KW
                                         + kw)
                                wv = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)
                                # outer: [TILE_P, 1] * [1, BLOCK_OC]
                                acc += xv[:, None] * wv[None, :]

                # add bias, divide
                bv = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0)
                acc = (acc + bv[None, :]) / divisor

                # max-pool update
                maxv = tl.maximum(maxv, acc)

    # store maxv: shape [TILE_P, BLOCK_OC] into out[N, OC, PD, PH, PW]
    # output layout: [N, OC, PD*PH*PW]
    out_idx = (pid_n * OC * p_total
               + oc_offsets[None, :] * p_total
               + p_offsets[:, None])
    store_mask = p_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_idx, maxv, mask=store_mask)


def fused_conv_div_maxpool(x, weight, bias, divisor, pool_size):
    N, IC, D, H, W = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = D - KD + 1
    OH = H - KH + 1
    OW = W - KW + 1
    POOL_D, POOL_H, POOL_W = pool_size
    PD = OD // POOL_D
    PH = OH // POOL_H
    PW = OW // POOL_W

    # output shape [N, OC, PD, PH, PW]
    out = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=torch.float32)

    TILE_P = 32
    BLOCK_OC = 16  # OC = 16

    p_total = PD * PH * PW
    grid = (N, (p_total + TILE_P - 1) // TILE_P)

    fused_conv_pool_kernel[grid](
        x, weight, bias, out,
        N, IC, D, H, W,
        OC, OD, OH, OW,
        PD, PH, PW,
        float(divisor),
        BLOCK_OC=BLOCK_OC,
        KD=KD, KH=KH, KW=KW,
        POOL_D=POOL_D, POOL_H=POOL_H, POOL_W=POOL_W,
        TILE_P=TILE_P,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.pool_size = pool_size
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        # fused conv + div + maxpool
        pooled = fused_conv_div_maxpool(x, w, b, self.divisor, self.pool_size)
        # pooled: [N, OC, PD, PH, PW]
        # global avg pool over spatial -> [N, OC, 1, 1, 1]
        N, OC = pooled.shape[0], pooled.shape[1]
        gap = pooled.view(N, OC, -1).mean(dim=2).view(N, OC, 1, 1, 1)
        # add bias, sum along sum_dim
        out = gap + self.bias
        out = torch.sum(out, dim=self.sum_dim)
        return out