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
    BLOCK_K: tl.constexpr,  # IC*KD*KH*KW tile
):
    pid_n = tl.program_id(0)         # batch
    pid_oc = tl.program_id(1)        # OC tile
    pid_sp = tl.program_id(2)        # spatial tile

    OS = OD * OH * OW

    offs_oc = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_sp = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    oc_mask = offs_oc < OC
    sp_mask = offs_sp < OS

    # decode spatial -> od, oh, ow
    od = offs_sp // (OH * OW)
    rem = offs_sp - od * (OH * OW)
    oh = rem // OW
    ow = rem - oh * OW

    K = IC * KD * KH * KW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # input base for this batch
    x_batch_base = pid_n * IC * ID * IH * IW

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = offs_k < K

        # decode k -> ic, kd, kh, kw
        kw_i = offs_k % KW
        tmp1 = offs_k // KW
        kh_i = tmp1 % KH
        tmp2 = tmp1 // KH
        kd_i = tmp2 % KD
        ic_i = tmp2 // KD

        # Load weight tile [BLOCK_M, BLOCK_K]
        # weight layout: [OC, IC, KD, KH, KW] -> oc * (IC*KD*KH*KW) + k
        w_offsets = offs_oc[:, None] * K + offs_k[None, :]
        w_mask = oc_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        # Build input offsets [BLOCK_K, BLOCK_N]
        # for each (k, sp): id = od + kd, ih = oh + kh, iw = ow + kw
        id_idx = od[None, :] + kd_i[:, None]   # [BLOCK_K, BLOCK_N]
        ih_idx = oh[None, :] + kh_i[:, None]
        iw_idx = ow[None, :] + kw_i[:, None]
        ic_idx = ic_i[:, None]                  # [BLOCK_K, 1] broadcast

        x_offsets = x_batch_base + ic_idx * (ID * IH * IW) + id_idx * (IH * IW) + ih_idx * IW + iw_idx
        x_mask = k_mask[:, None] & sp_mask[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        acc += tl.dot(w_tile, x_tile)

    # add bias
    bias = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]

    # store [BLOCK_M, BLOCK_N] to out[N, OC, OS]
    out_base = pid_n * OC * OS
    out_offsets = out_base + offs_oc[:, None] * OS + offs_sp[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


def conv3d_triton(x, weight, bias):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_M = 16   # OC tile (OC=16)
    BLOCK_N = 128  # spatial tile
    BLOCK_K = 32   # K tile (K = 3*3*3*3 = 81)

    OS = OD * OH * OW
    grid = (N, triton.cdiv(OC, BLOCK_M), triton.cdiv(OS, BLOCK_N))

    conv3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
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

    BLOCK_SIZE = 1024
    grid = (N * groups,)
    fused_gn_min_clamp_kernel[grid](
        x, out, weight, bias,
        N, C, S, groups, channels_per_group,
        float(min_value), float(max_value), float(eps),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
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