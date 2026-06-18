import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, D, H, W,
    OC, OD, OH, OW,
    KD, KH, KW,
    BLOCK_W: tl.constexpr,
    OC_BLOCK: tl.constexpr,
):
    # grid: (N * OD * OH, num_w_tiles, OC tiles)
    pid_ndh = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_oc = tl.program_id(2)

    n = pid_ndh // (OD * OH)
    rem = pid_ndh % (OD * OH)
    od = rem // OH
    oh = rem % OH

    ow_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    ow_mask = ow_offs < OW

    oc_offs = pid_oc * OC_BLOCK + tl.arange(0, OC_BLOCK)
    oc_mask = oc_offs < OC

    # Load bias for this OC tile
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    # acc: [OC_BLOCK, BLOCK_W]
    acc = bias[:, None] + tl.zeros((OC_BLOCK, BLOCK_W), dtype=tl.float32)

    # Loop over kernel and input channels (small: 3*3*3*3 = 81)
    for ic in tl.static_range(0, 3):
        for kd in tl.static_range(0, 3):
            id_ = od + kd
            for kh in tl.static_range(0, 3):
                ih = oh + kh
                for kw in tl.static_range(0, 3):
                    iw = ow_offs + kw
                    # x: [BLOCK_W]
                    x_idx = ((n * IC + ic) * D + id_) * H * W + ih * W + iw
                    x_val = tl.load(x_ptr + x_idx, mask=ow_mask, other=0.0)
                    # w: [OC_BLOCK]
                    w_idx = ((oc_offs * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)
                    acc += w_val[:, None] * x_val[None, :]

    # Store output: [OC_BLOCK, BLOCK_W]
    out_idx = ((n * OC + oc_offs[:, None]) * OD + od) * OH * OW + oh * OW + ow_offs[None, :]
    mask = oc_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=mask)


def triton_conv3d(x, weight, bias):
    N, IC, D, H, W = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = D - KD + 1
    OH = H - KH + 1
    OW = W - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_W = 64
    OC_BLOCK = 16  # OC is 16

    grid = (N * OD * OH, (OW + BLOCK_W - 1) // BLOCK_W, (OC + OC_BLOCK - 1) // OC_BLOCK)
    conv3d_kernel[grid](
        x, weight, bias, out,
        N, IC, D, H, W,
        OC, OD, OH, OW,
        KD, KH, KW,
        BLOCK_W=BLOCK_W,
        OC_BLOCK=OC_BLOCK,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def fused_gn_min_clamp_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    S, CPG,
    min_value, max_value, eps,
    stride_n, stride_g,
    BLOCK_S: tl.constexpr,
    CPG_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    base = pid_n * stride_n + pid_g * stride_g
    inv_count = 1.0 / (CPG * S)

    sum_val = 0.0
    sum_sq = 0.0

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        for c_idx in tl.static_range(0, CPG_C):
            c_offset = base + c_idx * S
            vals = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0)
            sum_val += tl.sum(vals, axis=0)
            sum_sq += tl.sum(vals * vals, axis=0)

    mean = sum_val * inv_count
    var = sum_sq * inv_count - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for c_idx in tl.static_range(0, CPG_C):
        channel = pid_g * CPG_C + c_idx
        w = tl.load(weight_ptr + channel)
        b = tl.load(bias_ptr + channel)
        scale = rstd * w
        shift = b - mean * scale
        c_offset = base + c_idx * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            vals = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0)
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

    BLOCK_S = 2048
    CPG_C = CPG

    stride_n = C * S
    stride_g = CPG * S

    grid = (N, groups)
    fused_gn_min_clamp_kernel[grid](
        x_flat, out, weight, bias,
        S, CPG,
        float(min_value), float(max_value), float(eps),
        stride_n, stride_g,
        BLOCK_S=BLOCK_S,
        CPG_C=CPG_C,
        num_warps=8,
        num_stages=2,
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
        x = triton_conv3d(x, self.conv.weight, self.conv.bias)
        x = fused_gn_min_clamp(
            x, self.norm.weight, self.norm.bias,
            self.groups, self.min_value, self.max_value, self.eps
        )
        x = self.dropout(x)
        return x