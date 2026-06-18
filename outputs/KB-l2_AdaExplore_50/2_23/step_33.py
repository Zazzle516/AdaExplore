import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_S: tl.constexpr,
):
    # grid: (N, ceil(OC/BLOCK_OC), ceil(OD*OH*OW/BLOCK_S))
    n = tl.program_id(0)
    oc_block = tl.program_id(1)
    s_block = tl.program_id(2)

    OS = OD * OH * OW
    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)      # [BLOCK_S]
    oc_mask = oc_offs < OC
    s_mask = s_offs < OS

    # decode s -> (od, oh, ow)
    od = s_offs // (OH * OW)
    rem = s_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros([BLOCK_OC, BLOCK_S], dtype=tl.float32)

    K = IC * KD * KH * KW

    # iterate K dimension
    for k in tl.static_range(0, KD * KH * KW):
        kd = k // (KH * KW)
        kr = k % (KH * KW)
        kh = kr // KW
        kw = kr % KW

        id_ = od + kd  # [BLOCK_S]
        ih = oh + kh
        iw = ow + kw

        for ic in range(0, IC):
            # load weight: w[oc, ic, kd, kh, kw]
            w_off = oc_offs * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
            w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

            # load input: x[n, ic, id_, ih, iw]
            x_off = n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + id_ * (IH * IW) + ih * IW + iw
            x_val = tl.load(x_ptr + x_off, mask=s_mask, other=0.0)  # [BLOCK_S]

            acc += w_val[:, None] * x_val[None, :]

    # add bias
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_val[:, None]

    # store: out[n, oc, od, oh, ow]
    out_off = n * (OC * OS) + oc_offs[:, None] * OS + s_offs[None, :]
    mask = oc_mask[:, None] & s_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


@triton.jit
def group_norm_mean_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    G, CPG, S,
    eps: tl.constexpr,
    inv_total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    group_size = CPG * S
    base = n * (G * CPG * S) + g * CPG * S

    sum_v = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_v += v
        sum_sq += v * v
    s_v = tl.sum(sum_v, axis=0)
    s_sq = tl.sum(sum_sq, axis=0)
    inv_gs = 1.0 / group_size
    mean = s_v * inv_gs
    var = s_sq * inv_gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        c_local = idx // S
        c_global = g * CPG + c_local
        w = tl.load(w_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(b_ptr + c_global, mask=mask, other=0.0)
        y = (v - mean) * rstd * w + b
        y = tl.where(mask, y, 0.0)
        acc += y
    group_sum = tl.sum(acc, axis=0)
    tl.atomic_add(out_ptr + n, group_sum * inv_total)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_S = 128
        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OD * OH * OW, BLOCK_S))
        conv3d_kernel[grid](
            x, self.conv.weight, self.conv.bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            BLOCK_OC=BLOCK_OC, BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2,
        )

        C = OC
        S = OD * OH * OW
        G = self.num_groups
        CPG = C // G
        total = C * S
        inv_total = 1.0 / total

        out = torch.zeros(N, device=x.device, dtype=x.dtype)
        BLOCK = 2048
        grid2 = (N * G,)
        group_norm_mean_kernel[grid2](
            conv_out, self.group_norm.weight, self.group_norm.bias, out,
            G, CPG, S,
            self.eps, inv_total,
            BLOCK=BLOCK, num_warps=8, num_stages=3,
        )
        return out