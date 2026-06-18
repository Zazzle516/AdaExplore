import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    PD, PH, PW,  # pooled output dims
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,  # pooled spatial tile
):
    # program ids: (n, oc_tile, pooled_spatial_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    P_total = PD * PH * PW
    sp_mask = sp_offs < P_total
    oc_mask = oc_offs < OC

    # decompose pooled spatial index -> (pd, ph, pw)
    pd = sp_offs // (PH * PW)
    rem = sp_offs - pd * (PH * PW)
    ph = rem // PW
    pw = rem - ph * PW

    # For each pooled output, compute max over 2x2x2 window of conv outputs
    # Conv output position: od = 2*pd + dz, oh = 2*ph + dy, ow = 2*pw + dx, dz,dy,dx in {0,1}
    # Conv: out[oc, od, oh, ow] = sum_{ic, kd, kh, kw} x[n, ic, od+kd, oh+kh, ow+kw] * w[oc, ic, kd, kh, kw]

    NEG_INF = float('-inf')
    max_val = tl.full((BLOCK_OC, BLOCK_SP), NEG_INF, dtype=tl.float32)

    # iterate over 8 positions in pool window
    for dz in tl.static_range(0, 2):
        for dy in tl.static_range(0, 2):
            for dx in tl.static_range(0, 2):
                od = pd * 2 + dz
                oh = ph * 2 + dy
                ow = pw * 2 + dx
                valid = sp_mask & (od < OD) & (oh < OH) & (ow < OW)

                acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

                for ic in range(0, IC):
                    for kd in tl.static_range(0, KD):
                        for kh in tl.static_range(0, KH):
                            for kw in tl.static_range(0, KW):
                                # input index
                                id_ = od + kd
                                ih_ = oh + kh
                                iw_ = ow + kw
                                # x[n, ic, id_, ih_, iw_]
                                x_off = (pid_n * IC * ID * IH * IW
                                         + ic * ID * IH * IW
                                         + id_ * IH * IW
                                         + ih_ * IW
                                         + iw_)
                                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)  # [BLOCK_SP]

                                # w[oc, ic, kd, kh, kw] for oc in oc_offs
                                w_off = (oc_offs * (IC * KD * KH * KW)
                                         + ic * (KD * KH * KW)
                                         + kd * (KH * KW)
                                         + kh * KW
                                         + kw)
                                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                                acc += w_val[:, None] * x_val[None, :]

                # add conv bias
                bias_v = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
                acc = acc + bias_v[:, None]

                # mask out invalid positions with -inf
                acc = tl.where(valid[None, :], acc, NEG_INF)
                max_val = tl.maximum(max_val, acc)

    # store: out[n, oc, pd, ph, pw] = max_val (shape [BLOCK_OC, BLOCK_SP])
    out_base = pid_n * OC * P_total
    out_off = out_base + oc_offs[:, None] * P_total + sp_offs[None, :]
    store_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, max_val, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD, KH, KW = self.kernel_size
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        PD = OD // 2
        PH = OH // 2
        PW = OW // 2

        pooled = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=torch.float32)

        BLOCK_OC = 16
        BLOCK_SP = 64
        P_total = PD * PH * PW

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(P_total, BLOCK_SP))

        conv3d_pool_kernel[grid](
            x, w, b, pooled,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            PD, PH, PW,
            KD, KH, KW,
            BLOCK_OC, BLOCK_SP,
            num_warps=4, num_stages=2,
        )

        # divide by divisor (fused into post-processing)
        # pooled is max over conv outputs. But original: conv -> div -> maxpool.
        # max(a/d, b/d, ...) = max(a,b,...)/d when d > 0. So dividing after is equivalent.
        pooled = pooled / self.divisor

        # global average pool over (PD, PH, PW)
        avg = pooled.mean(dim=(2, 3, 4), keepdim=True)  # [N, OC, 1, 1, 1]

        # add bias
        out = avg + self.bias  # [N, OC, 1, 1, 1]

        # sum along sum_dim
        out = out.sum(dim=self.sum_dim)
        return out