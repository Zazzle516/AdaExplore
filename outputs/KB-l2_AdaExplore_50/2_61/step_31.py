import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_relu_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # grid: (N, ceil(OC/BLOCK_OC), ceil(OD*OH*OW/BLOCK_SP))
    n = tl.program_id(0)
    oc_block = tl.program_id(1)
    sp_block = tl.program_id(2)

    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = sp_block * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    OHW = OH * OW
    OSP = OD * OHW

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < OSP

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # ConvTranspose3d with stride=1, padding=0, no flip in our formulation:
    # out[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw} x[n, ic, od-kd, oh-kh, ow-kw] * w[ic, oc, kd, kh, kw]
    # where 0 <= od-kd < ID etc.
    # IC reduction outer; KD,KH,KW unrolled (constexpr).
    for kd in tl.static_range(0, KD):
        id_ = od - kd  # [BLOCK_SP]
        d_valid = (id_ >= 0) & (id_ < ID)
        for kh in tl.static_range(0, KH):
            ih_ = oh - kh
            h_valid = (ih_ >= 0) & (ih_ < IH)
            for kw in tl.static_range(0, KW):
                iw_ = ow - kw
                w_valid = (iw_ >= 0) & (iw_ < IW)
                spatial_valid = d_valid & h_valid & w_valid & sp_mask  # [BLOCK_SP]

                # input offsets per spatial position (per ic stride added in loop)
                in_sp_offset = id_ * IH * IW + ih_ * IW + iw_  # [BLOCK_SP]

                for ic in range(0, IC):
                    x_off = n * IC * ID * IH * IW + ic * ID * IH * IW + in_sp_offset
                    x_val = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)  # [BLOCK_SP]

                    # weight: [IC, OC, KD, KH, KW]
                    w_off = ic * OC * KD * KH * KW + oc_offs * KD * KH * KW + kd * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += w_val[:, None] * x_val[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store: out[n, oc, od, oh, ow]
    out_off = n * OC * OSP + oc_offs[:, None] * OSP + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


@triton.jit
def fused_groupnorm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    group_size = C_PER_GROUP * S
    base = n * C * S + g * C_PER_GROUP * S

    sum_x = 0.0
    sum_x2 = 0.0

    for c in range(0, C_PER_GROUP):
        c_offset = base + c * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0)
            sum_x += tl.sum(tl.where(mask, v, 0.0), axis=0)
            sum_x2 += tl.sum(tl.where(mask, v * v, 0.0), axis=0)

    mean = sum_x / group_size
    var = sum_x2 / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for c in range(0, C_PER_GROUP):
        c_idx = g * C_PER_GROUP + c
        w = tl.load(weight_ptr + c_idx)
        b = tl.load(bias_ptr + c_idx)
        c_offset = base + c * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0)
            y = (v - mean) * rstd * w + b
            tl.store(out_ptr + c_offset + offs, y, mask=mask)


def conv_transpose3d_relu(x, w):
    # x: [N, IC, ID, IH, IW]
    # w: [IC, OC, KD, KH, KW]
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = w.shape
    OD = ID + KD - 1
    OH = IH + KH - 1
    OW = IW + KW - 1

    x = x.contiguous()
    w = w.contiguous()
    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 128
    OSP = OD * OH * OW

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OSP, BLOCK_SP))
    conv_transpose3d_relu_kernel[grid](
        x, w, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD=KD, KH=KH, KW=KW,
        BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
        num_warps=4, num_stages=2,
    )
    return out


def fused_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    C_PER_GROUP = C // groups
    x_flat = x.contiguous().view(N, C, S)
    out = torch.empty_like(x_flat)

    BLOCK_S = 1024
    grid = (N * groups,)
    fused_groupnorm_kernel[grid](
        x_flat, out, weight, bias,
        N, C, S,
        GROUPS=groups,
        C_PER_GROUP=C_PER_GROUP,
        eps=eps,
        BLOCK_S=BLOCK_S,
        num_warps=4,
    )
    return out.view(N, C, D, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.groups = groups
        self.eps = 1e-5

    def forward(self, x):
        y = conv_transpose3d_relu(x, self.conv_transpose.weight)
        y = fused_groupnorm(y, self.group_norm.weight, self.group_norm.bias, self.groups, self.eps)
        return y