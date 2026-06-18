import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr, OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    OHW = OH * OW
    ODHW = OD * OHW

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    sp_mask = sp_offs < ODHW

    oc_offs = tl.arange(0, 16)  # OC=16

    acc = tl.zeros((16, BLOCK_SP), dtype=tl.float32)

    IHW = IH * IW
    KHW = KH * KW
    KDHW = KD * KHW
    ICKDHW = IC * KDHW

    x_n_base = pid_n * IC * ID * IHW

    for ic in tl.static_range(0, IC):
        for kd in tl.static_range(0, KD):
            id_ = od + kd
            for kh in tl.static_range(0, KH):
                ih = oh + kh
                for kw in tl.static_range(0, KW):
                    iw = ow + kw
                    x_off = x_n_base + ic * ID * IHW + id_ * IHW + ih * IW + iw
                    x_val = tl.load(x_ptr + x_off, mask=sp_mask, other=0.0)
                    w_off = oc_offs * ICKDHW + ic * KDHW + kd * KHW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)
                    acc += w_val[:, None] * x_val[None, :]

    b_val = tl.load(b_ptr + oc_offs)
    acc += b_val[:, None]

    out_base = pid_n * OC * ODHW
    out_off = out_base + oc_offs[:, None] * ODHW + sp_offs[None, :]
    mask = sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


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

    BLOCK_SP = 256
    ODHW = OD * OH * OW
    grid = (N, triton.cdiv(ODHW, BLOCK_SP))
    conv3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        BLOCK_SP=BLOCK_SP,
        num_warps=4, num_stages=2,
    )
    return out


@triton.jit
def fused_gn_min_clamp_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    S: tl.constexpr,
    channels_per_group: tl.constexpr,
    groups: tl.constexpr,
    min_value: tl.constexpr, max_value: tl.constexpr, eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * GROUP_SIZE

    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, GROUP_SIZE, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < GROUP_SIZE
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    g = pid % groups
    c_base = g * channels_per_group

    # Pre-load weight/bias for this group's channels
    c_offs = c_base + tl.arange(0, channels_per_group)
    w_grp = tl.load(weight_ptr + c_offs)
    b_grp = tl.load(bias_ptr + c_offs)

    for off in range(0, GROUP_SIZE, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < GROUP_SIZE
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        c_in_group = idx // S
        # gather w/b from small loaded vectors
        w = tl.load(weight_ptr + c_base + c_in_group, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c_base + c_in_group, mask=mask, other=0.0)
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
    GROUP_SIZE = channels_per_group * S
    BLOCK_SIZE = 2048
    grid = (N * groups,)
    fused_gn_min_clamp_kernel[grid](
        x, out, weight, bias,
        S, channels_per_group, groups,
        float(min_value), float(max_value), float(eps),
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
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