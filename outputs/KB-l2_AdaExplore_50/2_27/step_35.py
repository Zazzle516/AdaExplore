import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SP': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 512}, num_warps=8, num_stages=2),
    ],
    key=['B', 'OC', 'OD', 'OH', 'OW', 'IC_CONST', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv3d_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_CONST: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
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
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    # grid: (B, NUM_GROUPS)
    # Single-pass: accumulate group sum/sum_sq AND per-channel sum simultaneously
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    GROUP_SIZE = CHANNELS_PER_GROUP * S

    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)
    # per-channel sums stored as separate scalars via accumulator vector
    # We'll use a vector of size CHANNELS_PER_GROUP
    chan_sums = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)

    # Streaming pass over S
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        for c_local in tl.static_range(CHANNELS_PER_GROUP):
            c = pid_g * CHANNELS_PER_GROUP + c_local
            base = (pid_b * C + c) * S
            v = tl.load(x_ptr + base + s_offs, mask=mask, other=0.0).to(tl.float32)
            s_v = tl.sum(v, axis=0)
            s_v2 = tl.sum(v * v, axis=0)
            sum_x += s_v
            sum_x2 += s_v2
            # accumulate per-channel sum into chan_sums[c_local]
            # use one-hot trick to avoid scalar indexing
            mask_vec = tl.arange(0, CHANNELS_PER_GROUP) == c_local
            chan_sums += tl.where(mask_vec, s_v, 0.0)

    mean = sum_x / GROUP_SIZE
    var = sum_x2 / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    inv_S = 1.0 / S

    # Load gamma/beta for this group's channels
    c_idx = pid_g * CHANNELS_PER_GROUP + tl.arange(0, CHANNELS_PER_GROUP)
    gamma = tl.load(gamma_ptr + c_idx).to(tl.float32)
    beta = tl.load(beta_ptr + c_idx).to(tl.float32)

    # chan_mean_raw = chan_sums / S
    # output = (chan_mean_raw - mean) * rstd * gamma + beta
    chan_mean_raw = chan_sums * inv_S
    out_vals = (chan_mean_raw - mean) * rstd * gamma + beta

    out_offs = pid_b * C + c_idx
    tl.store(out_ptr + out_offs, out_vals)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups

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

        grid = lambda meta: (B, OC, (OSP + meta['BLOCK_SP'] - 1) // meta['BLOCK_SP'])
        conv3d_hardswish_kernel[grid](
            x, self.weight, self.bias, conv_out,
            B, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            IC,
        )

        out = torch.empty((B, OC), device=x.device, dtype=torch.float32)
        CPG = OC // self.num_groups

        BLOCK_S = 1024
        if OSP < BLOCK_S:
            BLOCK_S = 1
            while BLOCK_S < OSP:
                BLOCK_S *= 2

        grid2 = (B, self.num_groups)
        groupnorm_mean_kernel[grid2](
            conv_out, self.gn_weight, self.gn_bias, out,
            B, OC, OSP,
            self.num_groups, CPG,
            BLOCK_S,
            self.eps,
            num_warps=4,
            num_stages=2,
        )
        return out