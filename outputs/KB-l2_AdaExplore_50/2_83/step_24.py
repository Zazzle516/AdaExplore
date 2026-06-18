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
    IC_C: tl.constexpr, OC_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # program: (n, s_tile)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    OS = OD * OH * OW
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < OS

    # decode s -> (od, oh, ow)
    ow = s_offs % OW
    oh = (s_offs // OW) % OH
    od = s_offs // (OH * OW)

    # accumulator [OC_C, BLOCK_S]
    acc = tl.zeros((OC_C, BLOCK_S), dtype=tl.float32)

    # input base for this batch
    x_n_base = pid_n * IC * ID * IH * IW

    for ic in tl.static_range(0, IC_C):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw
                    in_idx = x_n_base + ic * (ID * IH * IW) + id_ * (IH * IW) + ih_ * IW + iw_
                    x_val = tl.load(x_ptr + in_idx, mask=s_mask, other=0.0)  # [BLOCK_S]

                    # weight: [OC, IC, KD, KH, KW]
                    oc_range = tl.arange(0, OC_C)
                    w_idx = oc_range * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_idx)  # [OC_C]

                    acc += w_val[:, None] * x_val[None, :]

    # add bias
    oc_range = tl.arange(0, OC_C)
    bias = tl.load(b_ptr + oc_range)
    acc += bias[:, None]

    # store: out[n, oc, od, oh, ow], shape [N, OC, OD, OH, OW]
    out_n_base = pid_n * OC * OS
    out_idx = out_n_base + oc_range[:, None] * OS + s_offs[None, :]
    out_mask = s_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


@triton.jit
def groupnorm_min_clamp_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, S, G, CPG,
    EPS: tl.constexpr,
    MIN_VAL: tl.constexpr,
    MAX_VAL: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    base = n * C * S + g * CPG * S
    inv_count = 1.0 / (CPG * S)

    sum_val = 0.0
    sum_sq = 0.0

    for c in range(0, CPG):
        ch_base = base + c * S
        for off in range(0, S, BLOCK_S):
            idx = off + tl.arange(0, BLOCK_S)
            mask = idx < S
            vals = tl.load(x_ptr + ch_base + idx, mask=mask, other=0.0)
            sum_val += tl.sum(vals, axis=0)
            sum_sq += tl.sum(vals * vals, axis=0)

    mean = sum_val * inv_count
    var = sum_sq * inv_count - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)

    for c in range(0, CPG):
        c_global = g * CPG + c
        w = tl.load(weight_ptr + c_global)
        b = tl.load(bias_ptr + c_global)
        scale = rstd * w
        shift = b - mean * scale
        ch_base = base + c * S
        for off in range(0, S, BLOCK_S):
            idx = off + tl.arange(0, BLOCK_S)
            mask = idx < S
            vals = tl.load(x_ptr + ch_base + idx, mask=mask, other=0.0)
            normalized = vals * scale + shift
            y = tl.minimum(normalized, MIN_VAL)
            y = tl.maximum(y, MIN_VAL)
            y = tl.minimum(y, MAX_VAL)
            tl.store(out_ptr + ch_base + idx, y, mask=mask)


def triton_conv3d(x, weight, bias):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    OS = OD * OH * OW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_S = 128
    grid = (N, (OS + BLOCK_S - 1) // BLOCK_S)

    conv3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD=KD, KH=KH, KW=KW,
        IC_C=IC, OC_C=OC,
        BLOCK_S=BLOCK_S,
        num_warps=4, num_stages=2,
    )
    return out


def triton_groupnorm_fused(x, weight, bias, groups, eps, min_value, max_value):
    N, C = x.shape[0], x.shape[1]
    spatial = x.shape[2:]
    S = 1
    for s in spatial:
        S *= s
    x_flat = x.contiguous().view(N, C, S)
    out = torch.empty_like(x_flat)
    CPG = C // groups
    grid = (N * groups,)
    BLOCK_S = 4096
    groupnorm_min_clamp_kernel[grid](
        x_flat, weight, bias, out,
        N, C, S, groups, CPG,
        EPS=float(eps),
        MIN_VAL=float(min_value),
        MAX_VAL=float(max_value),
        BLOCK_S=BLOCK_S,
        num_warps=8, num_stages=2,
    )
    return out.view(x.shape)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.groups = groups
        self.min_value = min_value
        self.max_value = max_value
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        x = triton_conv3d(x, self.conv.weight, self.conv.bias)
        x = triton_groupnorm_fused(x, self.norm.weight, self.norm.bias,
                                    self.groups, self.eps,
                                    self.min_value, self.max_value)
        x = self.dropout(x)
        return x