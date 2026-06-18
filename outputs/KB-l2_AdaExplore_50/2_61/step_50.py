import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, y_ptr, gamma_ptr, beta_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)

    group_size = C_PER_GROUP * S
    base = n * C * S + g * C_PER_GROUP * S

    sum_val = tl.zeros([BLOCK_S], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK_S], dtype=tl.float32)

    offs_s = tl.arange(0, BLOCK_S)
    for c in range(0, C_PER_GROUP):
        c_off = base + c * S
        for s_start in range(0, S, BLOCK_S):
            s_idx = s_start + offs_s
            mask = s_idx < S
            x = tl.load(x_ptr + c_off + s_idx, mask=mask, other=0.0, cache_modifier=".ca")
            x = tl.maximum(x, 0.0)
            sum_val += x
            sum_sq += x * x

    total_sum = tl.sum(sum_val)
    total_sq = tl.sum(sum_sq)
    inv_n = 1.0 / group_size
    mean = total_sum * inv_n
    var = total_sq * inv_n - mean * mean
    rstd = tl.rsqrt(var + eps)

    for c in range(0, C_PER_GROUP):
        c_global = g * C_PER_GROUP + c
        gamma = tl.load(gamma_ptr + c_global)
        beta = tl.load(beta_ptr + c_global)
        c_off = base + c * S
        bias_term = beta - mean * rstd * gamma
        scale = rstd * gamma
        for s_start in range(0, S, BLOCK_S):
            s_idx = s_start + offs_s
            mask = s_idx < S
            x = tl.load(x_ptr + c_off + s_idx, mask=mask, other=0.0, cache_modifier=".ca")
            x = tl.maximum(x, 0.0)
            y = x * scale + bias_term
            tl.store(y_ptr + c_off + s_idx, y, mask=mask)


def fused_relu_groupnorm(x, gamma, beta, groups, eps=1e-5):
    assert x.is_cuda and x.is_contiguous()
    N, C = x.shape[0], x.shape[1]
    spatial = x.shape[2:]
    S = 1
    for d in spatial:
        S *= d
    C_per_group = C // groups
    y = torch.empty_like(x)

    # S = 34*34*34 = 39304. Use BLOCK_S that covers it in few iterations.
    BLOCK_S = 8192
    grid = (N, groups)
    fused_relu_groupnorm_kernel[grid](
        x, y, gamma, beta,
        N, C, S,
        GROUPS=groups,
        C_PER_GROUP=C_per_group,
        BLOCK_S=BLOCK_S,
        eps=eps,
        num_warps=16,
        num_stages=2,
    )
    return y


@triton.jit
def conv_transpose3d_relu_kernel(
    x_ptr, w_ptr, y_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_S: tl.constexpr,
    APPLY_RELU: tl.constexpr,
):
    # One program per (batch * od, oc_tile, hw_tile)
    pid_n_d = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    n = pid_n_d // OD
    od = pid_n_d % OD

    HW = OH * OW

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_oc = offs_oc < OC
    mask_hw = offs_hw < HW

    oh = offs_hw // OW
    ow = offs_hw % OW

    acc = tl.zeros([BLOCK_OC, BLOCK_S], dtype=tl.float32)

    # weight shape: (IC, OC, KD, KH, KW)
    # output[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw} x[n, ic, od-kd, oh-kh, ow-kw] * w[ic, oc, kd, kh, kw]
    # valid when 0 <= od-kd < ID etc.

    for kd in tl.static_range(0, KD):
        id_pos = od - kd
        valid_d = (id_pos >= 0) & (id_pos < ID)
        for kh in tl.static_range(0, KH):
            ih_pos = oh - kh
            valid_h = (ih_pos >= 0) & (ih_pos < IH)
            for kw in tl.static_range(0, KW):
                iw_pos = ow - kw
                valid_w = (iw_pos >= 0) & (iw_pos < IW)
                valid_hw = valid_h & valid_w & mask_hw  # [BLOCK_S]

                if valid_d:
                    # load weight slice [IC, BLOCK_OC] for this (kd, kh, kw)
                    # load input slice [IC, BLOCK_S] for this (kd, kh, kw)
                    # accumulate via dot product
                    in_spatial = id_pos * (IH * IW) + ih_pos * IW + iw_pos  # [BLOCK_S]
                    in_base = n * IC * ID * IH * IW
                    w_base = kd * KH * KW + kh * KW + kw  # offset within (KD,KH,KW)

                    for ic in range(0, IC):
                        # load weight[ic, offs_oc, kd, kh, kw]
                        w_off = ic * OC * KD * KH * KW + offs_oc * KD * KH * KW + w_base
                        w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                        x_off = in_base + ic * ID * IH * IW + in_spatial
                        x_val = tl.load(x_ptr + x_off, mask=valid_hw, other=0.0)  # [BLOCK_S]

                        acc += w_val[:, None] * x_val[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # store output
    out_base = n * OC * OD * OH * OW + offs_oc[:, None] * (OD * OH * OW) + od * HW + offs_hw[None, :]
    out_mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(y_ptr + out_base, acc, mask=out_mask)


def conv_transpose3d_relu(x, weight, bias=None):
    # x: (N, IC, ID, IH, IW)
    # weight: (IC, OC, KD, KH, KW)
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    OD = ID + KD - 1
    OH = IH + KH - 1
    OW = IW + KW - 1

    y = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_S = 128
    HW = OH * OW

    grid = (N * OD, triton.cdiv(OC, BLOCK_OC), triton.cdiv(HW, BLOCK_S))
    conv_transpose3d_relu_kernel[grid](
        x, weight, y,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_S=BLOCK_S,
        APPLY_RELU=True,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.groups = groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        # Use custom conv-transpose + relu fused kernel
        y = conv_transpose3d_relu(x, w)
        # Then group norm
        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        # GN expects already-relu'd input; reuse fused kernel but skip relu by clamping (already non-negative)
        out = fused_relu_groupnorm(y, gamma, beta, self.groups, self.eps)
        return out