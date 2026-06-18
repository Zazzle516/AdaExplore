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
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
    BLOCK_K: tl.constexpr,  # IC*KD*KH*KW tile (we'll do full)
    IC_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_oc = tl.program_id(1)  # OC tile
    pid_sp = tl.program_id(2)  # spatial tile

    OS = OD * OH * OW

    offs_oc = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_sp = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_oc = offs_oc < OC
    mask_sp = offs_sp < OS

    # decode spatial indices
    od = offs_sp // (OH * OW)
    rem = offs_sp - od * (OH * OW)
    oh = rem // OW
    ow = rem - oh * OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K = IC_CONST * KD * KH * KW

    # base pointer into x for this batch
    x_batch = x_ptr + pid_n * IC * ID * IH * IW

    for k in range(K):
        # decode k
        ic = k // (KD * KH * KW)
        krem = k - ic * (KD * KH * KW)
        kd = krem // (KH * KW)
        krem2 = krem - kd * (KH * KW)
        kh = krem2 // KW
        kw = krem2 - kh * KW

        id_ = od + kd
        ih_ = oh + kh
        iw_ = ow + kw

        x_offs = ic * (ID * IH * IW) + id_ * (IH * IW) + ih_ * IW + iw_
        x_vals = tl.load(x_batch + x_offs, mask=mask_sp, other=0.0)  # [BLOCK_N]

        w_offs = offs_oc * (IC_CONST * KD * KH * KW) + k  # [BLOCK_M]
        w_vals = tl.load(w_ptr + w_offs, mask=mask_oc, other=0.0)  # [BLOCK_M]

        acc += w_vals[:, None] * x_vals[None, :]

    # add bias
    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += b_vals[:, None]

    # store
    out_base = out_ptr + pid_n * OC * OS
    out_offs = offs_oc[:, None] * OS + offs_sp[None, :]
    out_mask = mask_oc[:, None] & mask_sp[None, :]
    tl.store(out_base + out_offs, acc, mask=out_mask)


def conv3d_triton(x, weight, bias):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    OS = OD * OH * OW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_M = 16
    BLOCK_N = 128

    grid = (N, triton.cdiv(OC, BLOCK_M), triton.cdiv(OS, BLOCK_N))
    conv3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=1,
        IC_CONST=IC,
        num_warps=4, num_stages=2,
    )
    return out


@triton.jit
def fused_gn_min_clamp_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    N, C, S,
    groups, channels_per_group,
    min_value, max_value, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // groups
    g = pid % groups

    group_size = channels_per_group * S
    base = n * C * S + g * channels_per_group * S

    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, group_size, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_size
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, group_size, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_size
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        c_in_group = idx // S
        c = g * channels_per_group + c_in_group
        w = tl.load(weight_ptr + c, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c, mask=mask, other=0.0)
        y = (x - mean) * rstd * w + b
        y = tl.minimum(y, min_value)
        y = tl.maximum(y, min_value)
        y = tl.minimum(y, max_value)
        tl.store(out_ptr + base + idx, y, mask=mask)


def fused_gn_min_clamp(x, weight, bias, groups, min_value, max_value, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    channels_per_group = C // groups
    x = x.contiguous()
    out = torch.empty_like(x)

    group_size = channels_per_group * S
    BLOCK_SIZE = 1024
    grid = (N * groups,)
    fused_gn_min_clamp_kernel[grid](
        x, out, weight, bias,
        N, C, S, groups, channels_per_group,
        float(min_value), float(max_value), float(eps),
        BLOCK_SIZE=BLOCK_SIZE,
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

    def forward(self, x):
        x = x.contiguous()
        x = conv3d_triton(x, self.conv.weight, self.conv.bias)
        x = fused_gn_min_clamp(
            x,
            self.norm.weight,
            self.norm.bias,
            self.groups,
            self.min_value,
            self.max_value,
            self.norm.eps,
        )
        x = self.dropout(x)
        return x