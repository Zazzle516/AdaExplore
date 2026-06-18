import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv3d_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_CONST: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # grid: (B, OC, ceil(OD*OH*OW / BLOCK_SP))
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    OHW = OH * OW
    OSP = OD * OHW
    sp_mask = sp_offs < OSP

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    bias = tl.load(b_ptr + pid_oc).to(tl.float32)
    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32) + bias

    # iterate over IC, KD, KH, KW
    for ic in tl.static_range(IC_CONST):
        for kd in tl.static_range(KD):
            for kh in tl.static_range(KH):
                for kw in tl.static_range(KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw
                    x_idx = (((pid_b * IC + ic) * ID + id_) * IH + ih_) * IW + iw_
                    w_idx = (((pid_oc * IC_CONST + ic) * KD + kd) * KH + kh) * KW + kw
                    xv = tl.load(x_ptr + x_idx, mask=sp_mask, other=0.0).to(tl.float32)
                    wv = tl.load(w_ptr + w_idx).to(tl.float32)
                    acc += xv * wv

    # hardswish: x * relu6(x+3) / 6
    t = acc + 3.0
    t = tl.minimum(tl.maximum(t, 0.0), 6.0)
    out = acc * t * (1.0 / 6.0)

    out_idx = ((pid_b * OC + pid_oc) * OSP) + sp_offs
    tl.store(out_ptr + out_idx, out, mask=sp_mask)


@triton.jit
def groupnorm_mean_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    B, C, S,
    NUM_GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    S_CONST: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    # grid: (B, NUM_GROUPS)
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    GROUP_SIZE = CHANNELS_PER_GROUP * S_CONST

    # compute mean and var over all elements in group
    sum_x = 0.0
    sum_x2 = 0.0
    for c_local in tl.static_range(CHANNELS_PER_GROUP):
        c = pid_g * CHANNELS_PER_GROUP + c_local
        base = (pid_b * C + c) * S
        for s_start in range(0, S_CONST, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            mask = s_offs < S
            v = tl.load(x_ptr + base + s_offs, mask=mask, other=0.0).to(tl.float32)
            sum_x += tl.sum(v, axis=0)
            sum_x2 += tl.sum(v * v, axis=0)

    mean = sum_x / GROUP_SIZE
    var = sum_x2 / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Now compute per-channel mean of normalized output
    inv_S = 1.0 / S
    for c_local in tl.static_range(CHANNELS_PER_GROUP):
        c = pid_g * CHANNELS_PER_GROUP + c_local
        gamma = tl.load(gamma_ptr + c).to(tl.float32)
        beta = tl.load(beta_ptr + c).to(tl.float32)
        base = (pid_b * C + c) * S
        chan_sum = 0.0
        for s_start in range(0, S_CONST, BLOCK_S):
            s_offs = s_start + tl.arange(0, BLOCK_S)
            mask = s_offs < S
            v = tl.load(x_ptr + base + s_offs, mask=mask, other=0.0).to(tl.float32)
            n = (v - mean) * rstd * gamma + beta
            n = tl.where(mask, n, 0.0)
            chan_sum += tl.sum(n, axis=0)
        chan_mean = chan_sum * inv_S
        tl.store(out_ptr + pid_b * C + c, chan_mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups

        # Conv3d weight/bias
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.weight = nn.Parameter(conv.weight.data.clone())
        if bias:
            self.bias = nn.Parameter(conv.bias.data.clone())
        else:
            self.bias = nn.Parameter(torch.zeros(out_channels))

        gn = nn.GroupNorm(num_groups, out_channels)
        self.gn_weight = nn.Parameter(gn.weight.data.clone())
        self.gn_bias = nn.Parameter(gn.bias.data.clone())

        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous().cuda()
        B, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        OSP = OD * OH * OW

        conv_out = torch.empty((B, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_SP = 128
        grid = (B, OC, (OSP + BLOCK_SP - 1) // BLOCK_SP)
        conv3d_hardswish_kernel[grid](
            x, self.weight, self.bias, conv_out,
            B, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            IC,
            BLOCK_SP,
            num_warps=4,
        )

        out = torch.empty((B, OC), device=x.device, dtype=torch.float32)
        CPG = OC // self.num_groups
        # Find next pow2 >= OSP, capped
        BLOCK_S = 1
        while BLOCK_S < OSP and BLOCK_S < 4096:
            BLOCK_S *= 2
        if BLOCK_S > 1024:
            BLOCK_S = 1024

        grid2 = (B, self.num_groups)
        groupnorm_mean_kernel[grid2](
            conv_out, self.gn_weight, self.gn_bias, out,
            B, OC, OSP,
            self.num_groups, CPG, OSP,
            BLOCK_S,
            self.eps,
            num_warps=4,
        )
        return out