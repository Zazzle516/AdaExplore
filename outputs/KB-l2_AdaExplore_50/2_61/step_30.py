import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Custom ConvTranspose3d (kernel_size=3, stride=1, padding=0) with fused ReLU
# Implemented as gather/GEMM: one program per (N, OC_tile, output-spatial-tile).
# For each output point (od,oh,ow), gather input window
#   x[n, :, od-2..od, oh-2..oh, ow-2..ow] (valid positions only)
# and compute sum over (ic, kd, kh, kw) of x * weight[ic, oc, 2-kd, 2-kh, 2-kw].
# i.e. weight is "flipped" to match transposed-conv semantics.
# ---------------------------------------------------------------------------

@triton.jit
def conv_transpose3d_relu_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    sp_idx = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_idx < (OD * OH * OW)

    od = sp_idx // (OH * OW)
    rem = sp_idx % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_idx = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_idx < OC

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For each kernel position, gather contribution.
    # Output point (od,oh,ow) receives contribution from input (id,ih,iw) via
    # weight[ic, oc, kd, kh, kw] where id = od - kd, ih = oh - kh, iw = ow - kw
    # (with stride=1, padding=0). kd in [0..KD-1].
    for kd in tl.static_range(0, KD):
        id_ = od - kd  # [BLOCK_SP]
        d_valid = (id_ >= 0) & (id_ < ID)
        for kh in tl.static_range(0, KH):
            ih_ = oh - kh
            h_valid = (ih_ >= 0) & (ih_ < IH)
            for kw in tl.static_range(0, KW):
                iw_ = ow - kw
                w_valid = (iw_ >= 0) & (iw_ < IW)
                pos_mask = d_valid & h_valid & w_valid & sp_mask  # [BLOCK_SP]

                # Loop over input channels
                for ic in range(0, IC):
                    # x offset for this (n, ic, id_, ih_, iw_) -> [BLOCK_SP]
                    x_off = (
                        pid_n * (IC * ID * IH * IW)
                        + ic * (ID * IH * IW)
                        + id_ * (IH * IW)
                        + ih_ * IW
                        + iw_
                    )
                    x_val = tl.load(x_ptr + x_off, mask=pos_mask, other=0.0)  # [BLOCK_SP]

                    # weight[ic, oc, kd, kh, kw] -> [BLOCK_OC]
                    w_off = (
                        ic * (OC * KD * KH * KW)
                        + oc_idx * (KD * KH * KW)
                        + kd * (KH * KW)
                        + kh * KW
                        + kw
                    )
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += x_val[:, None] * w_val[None, :]

    # Fused ReLU
    acc = tl.maximum(acc, 0.0)

    out_off = (
        pid_n * (OC * OD * OH * OW)
        + oc_idx[None, :] * (OD * OH * OW)
        + sp_idx[:, None]
    )
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv_transpose3d_relu(x, weight):
    # x: [N, IC, ID, IH, IW]
    # weight: [IC, OC, KD, KH, KW]
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    OD = ID + KD - 1
    OH = IH + KH - 1
    OW = IW + KW - 1

    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_SP = 128

    grid = (
        N,
        triton.cdiv(OC, BLOCK_OC),
        triton.cdiv(OD * OH * OW, BLOCK_SP),
    )
    conv_transpose3d_relu_kernel[grid](
        x, weight, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out


# ---------------------------------------------------------------------------
# Fused GroupNorm kernel
# ---------------------------------------------------------------------------

@triton.jit
def groupnorm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    N, C, SPATIAL, CHANS_PER_GROUP,
    eps,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size = CHANS_PER_GROUP * SPATIAL
    base = pid_n * C * SPATIAL + pid_g * group_size

    sum_x = 0.0
    sum_x2 = 0.0
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_x += tl.sum(tl.where(mask, v, 0.0), axis=0)
        sum_x2 += tl.sum(tl.where(mask, v * v, 0.0), axis=0)

    inv = 1.0 / group_size
    mean = sum_x * inv
    var = sum_x2 * inv - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        c_in_group = idx // SPATIAL
        c_global = pid_g * CHANS_PER_GROUP + c_in_group
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0)
        y = (v - mean) * rstd * w + b
        tl.store(out_ptr + base + idx, y, mask=mask)


def fused_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    SPATIAL = D * H * W
    CHANS_PER_GROUP = C // groups
    x = x.contiguous()
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = (N, groups)
    groupnorm_kernel[grid](
        x, out, weight, bias,
        N, C, SPATIAL, CHANS_PER_GROUP,
        eps,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.groups = groups
        self.eps = 1e-5
        self.use_bias = bias

    def forward(self, x):
        x = x.contiguous()
        y = conv_transpose3d_relu(x, self.conv_transpose.weight)
        if self.use_bias and self.conv_transpose.bias is not None:
            # bias add + redo relu (rare path)
            y = y + self.conv_transpose.bias.view(1, -1, 1, 1, 1)
            y = torch.relu(y)
        y = fused_groupnorm(y, self.group_norm.weight, self.group_norm.bias, self.groups, self.eps)
        return y