import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_div_maxpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC, OD, OH, OW,
    PD, PH, PW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    POOL_D: tl.constexpr, POOL_H: tl.constexpr, POOL_W: tl.constexpr,
    inv_divisor,
    BLOCK_OC: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    P = PD * PH * PW
    sp_offs = pid_sp * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    sp_mask = sp_offs < P

    pd = sp_offs // (PH * PW)
    rem = sp_offs % (PH * PW)
    ph = rem // PW
    pw = rem % PW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # We need to compute max over POOL_D*POOL_H*POOL_W conv outputs for each (n, oc, pd, ph, pw)
    # For each pooled output, the conv outputs to consider are at:
    #   od = pd*POOL_D + i, oh = ph*POOL_H + j, ow = pw*POOL_W + k
    # for i in [0, POOL_D), j in [0, POOL_H), k in [0, POOL_W)
    
    NEG_INF = -3.4e38
    max_val = tl.full((BLOCK_OC, BLOCK_SPATIAL), NEG_INF, dtype=tl.float32)

    # Loop over pooling window
    for pi in tl.static_range(0, POOL_D):
        for pj in tl.static_range(0, POOL_H):
            for pk in tl.static_range(0, POOL_W):
                od = pd * POOL_D + pi  # [BLOCK_SPATIAL]
                oh = ph * POOL_H + pj
                ow = pw * POOL_W + pk

                # Compute conv output at (n, oc, od, oh, ow)
                acc = tl.zeros((BLOCK_OC, BLOCK_SPATIAL), dtype=tl.float32)

                for ic in range(0, IC):
                    for kd in tl.static_range(0, KD):
                        for kh in tl.static_range(0, KH):
                            for kw in tl.static_range(0, KW):
                                id_ = od + kd
                                ih = oh + kh
                                iw = ow + kw
                                # x: [N, IC, D, H, W]
                                x_off = pid_n * IC * D * H * W + ic * D * H * W + id_ * H * W + ih * W + iw
                                x_val = tl.load(x_ptr + x_off, mask=sp_mask, other=0.0)
                                # w: [OC, IC, KD, KH, KW]
                                w_off = oc_offs * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                acc += w_val[:, None] * x_val[None, :]

                # Add bias
                b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
                acc = acc + b_val[:, None]
                acc = acc * inv_divisor

                max_val = tl.maximum(max_val, acc)

    # Store: out [N, OC, PD, PH, PW]
    out_off = pid_n * OC * P + oc_offs[:, None] * P + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, max_val, mask=out_mask)


def fused_conv_div_maxpool(x, w, b, divisor, pool_size):
    N, IC, D, H, W = x.shape
    OC, _, KD, KH, KW = w.shape
    OD = D - KD + 1
    OH = H - KH + 1
    OW = W - KW + 1
    POOL_D, POOL_H, POOL_W = pool_size
    PD = OD // POOL_D
    PH = OH // POOL_H
    PW = OW // POOL_W

    out = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=torch.float32)

    BLOCK_OC = 16
    BLOCK_SPATIAL = 32
    P = PD * PH * PW

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(P, BLOCK_SPATIAL))
    fused_conv_div_maxpool_kernel[grid](
        x, w, b, out,
        N, IC, D, H, W,
        OC, OD, OH, OW,
        PD, PH, PW,
        KD, KH, KW,
        POOL_D, POOL_H, POOL_W,
        1.0 / divisor,
        BLOCK_OC=BLOCK_OC,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.pool_size = pool_size
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        pooled = fused_conv_div_maxpool(x, w, b, self.divisor, self.pool_size)
        # global avg pool -> (N, OC, 1, 1, 1)
        N, OC, PD, PH, PW = pooled.shape
        avg = pooled.mean(dim=(2, 3, 4), keepdim=True)
        avg = avg + self.bias
        out = avg.sum(dim=self.sum_dim)
        return out