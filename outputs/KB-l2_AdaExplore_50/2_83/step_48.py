import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------
# Custom Conv3d kernel (no padding, stride=1)
# Uses implicit GEMM-style tiling: one program per (N, output spatial block).
# Loads all OC weights inline since OC=16 is small.
# ---------------------------------------------------------------
@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    IC: tl.constexpr, OC: tl.constexpr,
    ID, IH, IW,
    OD, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    S,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offs < S

    ow = s_offs % OW
    tmp = s_offs // OW
    oh = tmp % OH
    od = tmp // OH

    IHW = IH * IW
    IDHW = ID * IHW
    KHW = KH * KW
    KDHW = KD * KHW
    ICKDHW = IC * KDHW

    acc = tl.zeros([BLOCK_S, OC], dtype=tl.float32)

    n_base = pid_n * IC * IDHW

    for ic in tl.static_range(0, IC):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw
                    x_off = n_base + ic * IDHW + id_ * IHW + ih_ * IW + iw_
                    x_val = tl.load(x_ptr + x_off, mask=mask_s, other=0.0)
                    w_off = ic * KDHW + kd * KHW + kh * KW + kw
                    w_idx = tl.arange(0, OC) * ICKDHW + w_off
                    w_val = tl.load(w_ptr + w_idx)
                    acc += x_val[:, None] * w_val[None, :]

    bias = tl.load(b_ptr + tl.arange(0, OC))
    acc += bias[None, :]

    oc_range = tl.arange(0, OC)
    out_offs = pid_n * (OC * S) + oc_range[None, :] * S + s_offs[:, None]
    out_mask = mask_s[:, None]
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
        IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        S,
        BLOCK_S=BLOCK_S,
        num_warps=4,
        num_stages=2,
    )
    return out


# ---------------------------------------------------------------
# Fused GroupNorm + min(min_value) + clamp(min_value, max_value)
# One program per (N, group)
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

    sum_val = tl.zeros([BLOCK_S], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK_S], dtype=tl.float32)

    # First pass: accumulate sums
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        for c_idx in tl.static_range(0, CPG):
            ptr = x_ptr + base + c_idx * S + offs
            vals = tl.load(ptr, mask=mask, other=0.0)
            sum_val += vals
            sum_sq += vals * vals

    total_sum = tl.sum(sum_val, axis=0)
    total_sumsq = tl.sum(sum_sq, axis=0)

    inv_count = 1.0 / (CPG * S)
    mean = total_sum * inv_count
    var = total_sumsq * inv_count - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        for c_idx in tl.static_range(0, CPG):
            channel = pid_g * CPG + c_idx
            w = tl.load(weight_ptr + channel)
            b = tl.load(bias_ptr + channel)
            ptr = x_ptr + base + c_idx * S + offs
            vals = tl.load(ptr, mask=mask, other=0.0)
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

    BLOCK_S = 4096

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
        self.dropout_p = dropout_p
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        x = conv3d_triton(x, self.conv.weight, self.conv.bias)
        x = fused_gn_min_clamp(
            x, self.norm.weight, self.norm.bias,
            self.groups, self.min_value, self.max_value, self.eps
        )
        if self.training and self.dropout_p > 0:
            x = self.dropout(x)
        return x