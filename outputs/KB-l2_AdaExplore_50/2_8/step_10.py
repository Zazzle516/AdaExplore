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
    PD, PH, PW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    inv_divisor,
    BLOCK_PH: tl.constexpr, BLOCK_PW: tl.constexpr,
):
    # program: (n*OC, pd, ph_block * pw_blocks + pw_block)
    pid_noc = tl.program_id(0)
    pid_pd = tl.program_id(1)
    pid_phw = tl.program_id(2)

    n = pid_noc // OC
    oc = pid_noc % OC

    num_pw_blocks = (PW + BLOCK_PW - 1) // BLOCK_PW
    pid_ph = pid_phw // num_pw_blocks
    pid_pw = pid_phw % num_pw_blocks

    ph_offs = pid_ph * BLOCK_PH + tl.arange(0, BLOCK_PH)
    pw_offs = pid_pw * BLOCK_PW + tl.arange(0, BLOCK_PW)
    ph_mask = ph_offs < PH
    pw_mask = pw_offs < PW

    # For each pooled output (pid_pd, ph_offs, pw_offs):
    # max over 2x2x2 window of conv outputs starting at (pid_pd*2, ph*2, pw*2)
    # conv output at (od, oh, ow) = sum over ic, kd, kh, kw of x[n, ic, od+kd, oh+kh, ow+kw] * w[oc, ic, kd, kh, kw]
    # then divide by divisor.

    neg_inf = float("-inf")
    max_val = tl.full((BLOCK_PH, BLOCK_PW), neg_inf, dtype=tl.float32)

    # iterate over 2x2x2 max-pool window
    for pkd in tl.static_range(0, 2):
        for pkh in tl.static_range(0, 2):
            for pkw in tl.static_range(0, 2):
                od = pid_pd * 2 + pkd
                oh_ = ph_offs * 2 + pkh  # [BLOCK_PH]
                ow_ = pw_offs * 2 + pkw  # [BLOCK_PW]

                od_valid = od < OD
                oh_valid = oh_ < OH  # [BLOCK_PH]
                ow_valid = ow_ < OW  # [BLOCK_PW]

                # compute conv output for this (od, oh_, ow_) tile
                acc = tl.zeros((BLOCK_PH, BLOCK_PW), dtype=tl.float32)

                for ic in range(0, IC):
                    for kd in tl.static_range(0, KD):
                        for kh in tl.static_range(0, KH):
                            for kw in tl.static_range(0, KW):
                                id_ = od + kd
                                ih_ = oh_[:, None] + kh  # [BLOCK_PH, 1]
                                iw_ = ow_[None, :] + kw  # [1, BLOCK_PW]

                                # load weight scalar
                                w_off = ((oc * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                                w_val = tl.load(w_ptr + w_off)

                                # load input
                                x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih_ * IW + iw_
                                x_mask = (ih_ < IH) & (iw_ < IW) & od_valid & oh_valid[:, None] & ow_valid[None, :]
                                x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                                acc += x_val * w_val

                # add bias
                bias_val = tl.load(b_ptr + oc)
                acc = (acc + bias_val) * inv_divisor

                # mask out-of-bounds with -inf
                valid = od_valid & oh_valid[:, None] & ow_valid[None, :]
                acc = tl.where(valid, acc, neg_inf)

                max_val = tl.maximum(max_val, acc)

    # store to intermediate pooled output [N, OC, PD, PH, PW]
    out_off = ((((n * OC) + oc) * PD + pid_pd) * PH + ph_offs[:, None]) * PW + pw_offs[None, :]
    out_mask = ph_mask[:, None] & pw_mask[None, :]
    tl.store(out_ptr + out_off, max_val, mask=out_mask)


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
        PD = OD // 2
        PH = OH // 2
        PW = OW // 2

        weight = self.conv.weight.contiguous()
        conv_bias = self.conv.bias.contiguous()

        pooled = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=torch.float32)

        BLOCK_PH = 8
        BLOCK_PW = 16
        num_pw_blocks = (PW + BLOCK_PW - 1) // BLOCK_PW
        num_ph_blocks = (PH + BLOCK_PH - 1) // BLOCK_PH

        grid = (N * OC, PD, num_ph_blocks * num_pw_blocks)

        conv3d_pool_kernel[grid](
            x, weight, conv_bias, pooled,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            PD, PH, PW,
            KD, KH, KW,
            1.0 / self.divisor,
            BLOCK_PH=BLOCK_PH, BLOCK_PW=BLOCK_PW,
            num_warps=4,
        )

        # global avg pool: mean over (PD, PH, PW)
        gap = pooled.mean(dim=(2, 3, 4), keepdim=True)  # [N, OC, 1, 1, 1]
        out = gap + self.bias
        out = torch.sum(out, dim=self.sum_dim)
        return out