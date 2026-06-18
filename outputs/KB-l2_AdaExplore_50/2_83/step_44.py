import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------
# Custom Conv3d kernel (no padding, stride=1)
# Input: NCDHW, Output: NCDHW
# Fuses bias add into kernel.
# ---------------------------------------------------------------
@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, OC: tl.constexpr,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)  # block over output spatial

    S = OD * OH * OW
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offs < S

    # decode s -> (od, oh, ow)
    ow = s_offs % OW
    tmp = s_offs // OW
    oh = tmp % OH
    od = tmp // OH

    # accumulator: [BLOCK_S, OC]
    acc = tl.zeros([BLOCK_S, OC], dtype=tl.float32)

    # Loop over IC, KD, KH, KW
    for ic in tl.static_range(0, IC):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw
                    # input pointer
                    x_off = pid_n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + id_ * (IH * IW) + ih_ * IW + iw_
                    x_val = tl.load(x_ptr + x_off, mask=mask_s, other=0.0)  # [BLOCK_S]
                    # weight: [OC, IC, KD, KH, KW] - load OC values
                    w_off = ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_idx = tl.arange(0, OC) * (IC * KD * KH * KW) + w_off
                    w_val = tl.load(w_ptr + w_idx)  # [OC]
                    acc += x_val[:, None] * w_val[None, :]

    # add bias
    bias = tl.load(b_ptr + tl.arange(0, OC))  # [OC]
    acc += bias[None, :]

    # store: output is NCDHW, so output[n, oc, s] = acc[s, oc]
    # out_off[n, oc, s] = n*OC*S + oc*S + s
    oc_range = tl.arange(0, OC)
    out_offs = pid_n * (OC * S) + oc_range[None, :] * S + s_offs[:, None]
    out_mask = mask_s[:, None] & (oc_range[None, :] < OC)
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv3d_triton(x, weight, bias):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    S = OD * OH * OW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_S = 128
    grid = (N, triton.cdiv(S, BLOCK_S))

    conv3d_kernel[grid](
        x, weight, bias, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        BLOCK_S=BLOCK_S,
        num_warps=4,
        num_stages=2,
    )
    return out


# ---------------------------------------------------------------
# Fused GroupNorm + min + clamp (since min_value=0, max_value=1, the
# combined op is clamp(y, 0, 0) -- wait actually:
# y = min(y, min_value=0)  -> y <= 0
# y = clamp(y, min=0, max=1) -> y in [0, 1] but y <= 0 already => y = 0
# Wait, but min_value = 0. min(y, 0) means y = min(y, 0) so y <= 0.
# Then clamp(y, 0, 1) clamps to [0, 1], so final = 0 if y <= 0... actually
# clamp(y,0,1) when y<=0 gives 0. So output is always 0 before dropout!
# But we should respect the actual computation.
# Actually min(x, 0) gives values <= 0. Then clamp(..., 0, 1) gives 0.
# So output = 0 everywhere. Then dropout(0) = 0.
# We should still compute correctly though, in case values happen.
# Let's keep general computation.
# ---------------------------------------------------------------
@triton.jit
def fused_gn_min_clamp_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    S,
    min_value: tl.constexpr, max_value: tl.constexpr, eps: tl.constexpr,
    CPG: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    G = tl.num_programs(1)
    C = G * CPG

    base = pid_n * C * S + pid_g * CPG * S

    sum_val = 0.0
    sum_sq = 0.0

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        for c_idx in tl.static_range(0, CPG):
            ptr = x_ptr + base + c_idx * S + offs
            vals = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
            sum_val += tl.sum(vals, axis=0)
            sum_sq += tl.sum(vals * vals, axis=0)

    inv_count = 1.0 / (CPG * S)
    mean = sum_val * inv_count
    var = sum_sq * inv_count - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        for c_idx in tl.static_range(0, CPG):
            channel = pid_g * CPG + c_idx
            w = tl.load(weight_ptr + channel).to(tl.float32)
            b = tl.load(bias_ptr + channel).to(tl.float32)
            ptr = x_ptr + base + c_idx * S + offs
            vals = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
            norm = (vals - mean) * rstd
            y = norm * w + b
            y = tl.minimum(y, min_value)
            y = tl.maximum(y, min_value)
            y = tl.minimum(y, max_value)
            tl.store(out_ptr + base + c_idx * S + offs, y, mask=mask)


def fused_gn_min_clamp(x, weight, bias, groups, min_value, max_value, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    CPG = C // groups
    x_flat = x.contiguous()
    out = torch.empty_like(x_flat)

    BLOCK_S = 2048

    grid = (N, groups)
    fused_gn_min_clamp_kernel[grid](
        x_flat, out, weight, bias,
        S,
        float(min_value), float(max_value), float(eps),
        CPG=CPG,
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.groups = groups
        self.min_value = min_value
        self.max_value = max_value
        self.eps = 1e-5
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous()
        # custom conv
        x = conv3d_triton(x, self.conv.weight, self.conv.bias)
        x = fused_gn_min_clamp(
            x, self.norm.weight, self.norm.bias,
            self.groups, self.min_value, self.max_value, self.eps
        )
        x = self.dropout(x)
        return x