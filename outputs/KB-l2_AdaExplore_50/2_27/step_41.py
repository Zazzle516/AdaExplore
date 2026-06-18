import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_hardswish_gn_mean_kernel(
    x_ptr,           # (B, IC, ID, IH, IW)
    w_ptr,           # (OC, IC, KD, KH, KW)
    conv_b_ptr,      # (OC,)
    gn_w_ptr,        # (OC,)
    gn_b_ptr,        # (OC,)
    out_ptr,         # (B, OC)
    B, IC,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    CPG: tl.constexpr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    c_start = pid_g * CPG
    c_offs = c_start + tl.arange(0, CPG)  # (CPG,)

    group_numel = CPG * S

    # store the hardswish(conv) intermediate into a buffer via re-computation in second pass
    # Pass 1: compute sum, sumsq of hs over (channels in group) x (spatial)
    sum_acc = tl.zeros((CPG,), dtype=tl.float32)
    sumsq_acc = tl.zeros((CPG,), dtype=tl.float32)

    s_range = tl.arange(0, BLOCK_S)

    # load conv bias for this group
    cb = tl.load(conv_b_ptr + c_offs).to(tl.float32)  # (CPG,)

    # Precompute number of spatial blocks
    # We iterate static range over S
    for s_start in tl.static_range(0, S, BLOCK_S):
        s_idx = s_start + s_range  # (BLOCK_S,)
        s_mask = s_idx < S

        # decode s_idx -> (od, oh, ow)
        od = s_idx // (OH * OW)
        rem = s_idx % (OH * OW)
        oh = rem // OW
        ow = rem % OW

        # accumulator (CPG, BLOCK_S)
        acc = tl.zeros((CPG, BLOCK_S), dtype=tl.float32)

        # Loop over IC, KD, KH, KW
        for ic in tl.static_range(0, IC):
            for kd in tl.static_range(0, KD):
                id_ = od + kd  # (BLOCK_S,)
                for kh in tl.static_range(0, KH):
                    ih = oh + kh
                    for kw in tl.static_range(0, KW):
                        iw = ow + kw
                        # load input: x[b, ic, id_, ih, iw]
                        in_off = (pid_b * IC * ID * IH * IW
                                  + ic * ID * IH * IW
                                  + id_ * IH * IW
                                  + ih * IW
                                  + iw)
                        xv = tl.load(x_ptr + in_off, mask=s_mask, other=0.0).to(tl.float32)  # (BLOCK_S,)
                        # load weight: w[c, ic, kd, kh, kw] for c in c_offs
                        w_off = (c_offs * (IC * KD * KH * KW)
                                 + ic * (KD * KH * KW)
                                 + kd * (KH * KW)
                                 + kh * KW
                                 + kw)
                        wv = tl.load(w_ptr + w_off).to(tl.float32)  # (CPG,)
                        acc += wv[:, None] * xv[None, :]

        # add bias
        acc = acc + cb[:, None]
        # hardswish
        t = acc + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        hs = acc * t * (1.0 / 6.0)
        mask2d = s_mask[None, :]
        hs = tl.where(mask2d, hs, 0.0)
        sum_acc += tl.sum(hs, axis=1)
        sumsq_acc += tl.sum(hs * hs, axis=1)

    grp_sum = tl.sum(sum_acc, axis=0)
    grp_sumsq = tl.sum(sumsq_acc, axis=0)
    mean = grp_sum / group_numel
    var = grp_sumsq / group_numel - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    gw = tl.load(gn_w_ptr + c_offs).to(tl.float32)  # (CPG,)
    gb = tl.load(gn_b_ptr + c_offs).to(tl.float32)

    # Pass 2: recompute conv, hardswish, then normalize + affine, accumulate mean over S
    out_acc = tl.zeros((CPG,), dtype=tl.float32)
    for s_start in tl.static_range(0, S, BLOCK_S):
        s_idx = s_start + s_range
        s_mask = s_idx < S
        od = s_idx // (OH * OW)
        rem = s_idx % (OH * OW)
        oh = rem // OW
        ow = rem % OW

        acc = tl.zeros((CPG, BLOCK_S), dtype=tl.float32)
        for ic in tl.static_range(0, IC):
            for kd in tl.static_range(0, KD):
                id_ = od + kd
                for kh in tl.static_range(0, KH):
                    ih = oh + kh
                    for kw in tl.static_range(0, KW):
                        iw = ow + kw
                        in_off = (pid_b * IC * ID * IH * IW
                                  + ic * ID * IH * IW
                                  + id_ * IH * IW
                                  + ih * IW
                                  + iw)
                        xv = tl.load(x_ptr + in_off, mask=s_mask, other=0.0).to(tl.float32)
                        w_off = (c_offs * (IC * KD * KH * KW)
                                 + ic * (KD * KH * KW)
                                 + kd * (KH * KW)
                                 + kh * KW
                                 + kw)
                        wv = tl.load(w_ptr + w_off).to(tl.float32)
                        acc += wv[:, None] * xv[None, :]

        acc = acc + cb[:, None]
        t = acc + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        hs = acc * t * (1.0 / 6.0)
        norm = (hs - mean) * rstd
        val = norm * gw[:, None] + gb[:, None]
        mask2d = s_mask[None, :]
        val = tl.where(mask2d, val, 0.0)
        out_acc += tl.sum(val, axis=1)

    out_mean = out_acc / S
    tl.store(out_ptr + pid_b * OC + c_offs, out_mean)


@triton.jit
def fused_post_conv_kernel(
    x_ptr,          # (B, C, S) input from conv
    out_ptr,        # (B, C) output
    gn_weight_ptr,  # (C,)
    gn_bias_ptr,    # (C,)
    B, C, S,
    CHANNELS_PER_GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    S_CONST: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    c_start = pid_g * CHANNELS_PER_GROUP
    c_offs = c_start + tl.arange(0, CHANNELS_PER_GROUP)

    group_numel = CHANNELS_PER_GROUP * S_CONST

    s_offs = tl.arange(0, BLOCK_S)

    sum_acc = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)
    sumsq_acc = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)

    for s_start in tl.static_range(0, S_CONST, BLOCK_S):
        s_idx = s_start + s_offs
        s_mask = s_idx < S_CONST
        ptrs = x_ptr + pid_b * C * S + c_offs[:, None] * S + s_idx[None, :]
        mask = s_mask[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        t = x + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        hs = x * t * (1.0 / 6.0)
        hs = tl.where(mask, hs, 0.0)
        sum_acc += tl.sum(hs, axis=1)
        sumsq_acc += tl.sum(hs * hs, axis=1)

    grp_sum = tl.sum(sum_acc, axis=0)
    grp_sumsq = tl.sum(sumsq_acc, axis=0)

    mean = grp_sum / group_numel
    var = grp_sumsq / group_numel - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(gn_weight_ptr + c_offs).to(tl.float32)
    b_aff = tl.load(gn_bias_ptr + c_offs).to(tl.float32)

    out_acc = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)
    for s_start in tl.static_range(0, S_CONST, BLOCK_S):
        s_idx = s_start + s_offs
        s_mask = s_idx < S_CONST
        ptrs = x_ptr + pid_b * C * S + c_offs[:, None] * S + s_idx[None, :]
        mask = s_mask[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        t = x + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        hs = x * t * (1.0 / 6.0)
        norm = (hs - mean) * rstd
        val = norm * w[:, None] + b_aff[:, None]
        val = tl.where(mask, val, 0.0)
        out_acc += tl.sum(val, axis=1)

    out_mean = out_acc / S_CONST
    tl.store(out_ptr + pid_b * C + c_offs, out_mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        B, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.reshape(B, C, S).contiguous()
        out = torch.empty((B, C), device=x.device, dtype=x.dtype)

        CHANNELS_PER_GROUP = C // self.num_groups
        BLOCK_S = 512

        grid = (B, self.num_groups)
        fused_post_conv_kernel[grid](
            x_flat, out,
            self.group_norm.weight, self.group_norm.bias,
            B, C, S,
            CHANNELS_PER_GROUP=CHANNELS_PER_GROUP,
            NUM_GROUPS=self.num_groups,
            S_CONST=S,
            BLOCK_S=BLOCK_S,
            eps=self.eps,
            num_warps=4,
        )
        return out