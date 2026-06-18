import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Conv3D kernel: implicit im2col GEMM
# One program per (N, OC_tile=full OC, spatial tile)
# ---------------------------------------------------------------------------
@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OS,  # OD*OH*OW
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # OC tile (= OC)
    IC_C: tl.constexpr,
):
    pid_n = tl.program_id(0)        # batch
    pid_m = tl.program_id(1)        # spatial tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_m = offs_m < OS

    # decompose spatial idx -> (od, oh, ow)
    od = offs_m // (OH * OW)
    rem = offs_m - od * (OH * OW)
    oh = rem // OW
    ow = rem - oh * OW

    offs_n = tl.arange(0, BLOCK_N)  # OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # input base for this batch
    x_batch = pid_n * IC * D * H * W

    # Loop over kD, kH, kW, IC
    for kd in tl.static_range(0, KD):
        id_ = od + kd  # input depth idx (pad=0, stride=1)
        for kh in tl.static_range(0, KH):
            ih = oh + kh
            for kw in tl.static_range(0, KW):
                iw = ow + kw
                # input offset for spatial part
                spatial_off = id_ * (H * W) + ih * W + iw  # [BLOCK_M]
                for ic in tl.static_range(0, IC_C):
                    # load x: [BLOCK_M]
                    x_off = x_batch + ic * (D * H * W) + spatial_off
                    x_val = tl.load(x_ptr + x_off, mask=mask_m, other=0.0)
                    # weight offset: w[oc, ic, kd, kh, kw], shape [OC, IC, KD, KH, KW]
                    w_off = offs_n * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)  # [BLOCK_N]
                    acc += x_val[:, None] * w_val[None, :]

    # add bias
    b_val = tl.load(b_ptr + offs_n)  # [BLOCK_N]
    acc += b_val[None, :]

    # store: output layout (N, OC, OS) contiguous
    out_off = pid_n * OC * OS + offs_n[None, :] * OS + offs_m[:, None]
    mask = mask_m[:, None]
    tl.store(out_ptr + out_off, acc, mask=mask)


def conv3d_triton(x, weight, bias):
    N, IC, D, H, W = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = D - KD + 1
    OH = H - KH + 1
    OW = W - KW + 1
    OS = OD * OH * OW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    BLOCK_N = OC  # 16
    IC_C = IC  # 3
    grid = (N, (OS + BLOCK_M - 1) // BLOCK_M)
    conv3d_kernel[grid](
        x, weight, bias, out,
        N, IC, D, H, W,
        OC, OD, OH, OW,
        KD, KH, KW,
        OS,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, IC_C=IC_C,
        num_warps=4, num_stages=2,
    )
    return out


# ---------------------------------------------------------------------------
# Fused GroupNorm + min + clamp
# ---------------------------------------------------------------------------
@triton.jit
def fused_gn_min_clamp_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    C, S, CPG,
    min_value, max_value, eps,
    BLOCK_S: tl.constexpr,
    CPG_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    base = pid_n * C * S + pid_g * CPG * S

    sum_val = 0.0
    sum_sq = 0.0

    for c_idx in tl.static_range(0, CPG_C):
        c_offset = base + c_idx * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            vals = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0)
            vals_f = vals.to(tl.float32)
            sum_val += tl.sum(tl.where(mask, vals_f, 0.0), axis=0)
            sum_sq += tl.sum(tl.where(mask, vals_f * vals_f, 0.0), axis=0)

    inv_count = 1.0 / (CPG * S)
    mean = sum_val * inv_count
    var = sum_sq * inv_count - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for c_idx in tl.static_range(0, CPG_C):
        channel = pid_g * CPG_C + c_idx
        w = tl.load(weight_ptr + channel).to(tl.float32)
        b = tl.load(bias_ptr + channel).to(tl.float32)
        # combine into scale/shift
        scale = w * rstd
        shift = b - mean * scale
        c_offset = base + c_idx * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            vals = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0).to(tl.float32)
            y = vals * scale + shift
            y = tl.minimum(y, min_value)
            y = tl.maximum(y, min_value)
            y = tl.minimum(y, max_value)
            tl.store(out_ptr + c_offset + offs, y, mask=mask)


def fused_gn_min_clamp(x, weight, bias, groups, min_value, max_value, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    CPG = C // groups
    x_flat = x.contiguous()
    out = torch.empty_like(x_flat)

    BLOCK_S = 1024
    grid = (N, groups)
    fused_gn_min_clamp_kernel[grid](
        x_flat, out, weight, bias,
        C, S, CPG,
        float(min_value), float(max_value), float(eps),
        BLOCK_S=BLOCK_S,
        CPG_C=CPG,
        num_warps=4,
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

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        x = conv3d_triton(x, w, b)
        x = fused_gn_min_clamp(
            x, self.norm.weight, self.norm.bias,
            self.groups, self.min_value, self.max_value, self.eps
        )
        x = self.dropout(x)
        return x