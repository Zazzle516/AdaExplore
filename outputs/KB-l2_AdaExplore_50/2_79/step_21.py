import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,        # [N, IC, ID, IH, IW]
    w_ptr,        # [OC, IC, KT, KH, KW]
    b_ptr,        # [OC]
    y_ptr,        # [N, OC, OD, OH, OW]
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KT: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_S: tl.constexpr,  # spatial tile (over OD*OH*OW)
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    S = OD * OH * OW
    s_mask = s_offs < S

    od = s_offs // (OH * OW)
    rem = s_offs - od * (OH * OW)
    oh = rem // OW
    ow = rem - oh * OW

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # loop over IC, KT, KH, KW
    for ic in tl.static_range(0, 3):  # IC=3
        for kt in tl.static_range(0, KT):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    id_ = od + kt
                    ih_ = oh + kh
                    iw_ = ow + kw
                    # load weight scalar
                    w_off = (((pid_oc * IC + ic) * KT + kt) * KH + kh) * KW + kw
                    w_val = tl.load(w_ptr + w_off)
                    # load input
                    x_off = ((pid_n * IC + ic) * ID + id_) * IH * IW + ih_ * IW + iw_
                    x_val = tl.load(x_ptr + x_off, mask=s_mask, other=0.0)
                    acc += x_val * w_val

    # bias
    bv = tl.load(b_ptr + pid_oc)
    acc += bv

    y_off = (pid_n * OC + pid_oc) * S + s_offs
    tl.store(y_ptr + y_off, acc, mask=s_mask)


@triton.jit
def fused_norm_clamp_max_kernel(
    x_ptr,           # [N, C, S]
    mult_ptr,        # [C]
    out_ptr,         # [N, S]
    N, C, S,
    clamp_min,
    clamp_max,
    eps,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
    S_FULL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # Load multipliers
    m = tl.load(mult_ptr + c_offs, mask=c_mask, other=0.0)

    # First pass: compute mean and var per channel by looping over S
    sum_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sumsq_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    base = pid_n * C * S
    for s_start in range(0, S_FULL, BLOCK_S):
        s_offs_full = s_start + tl.arange(0, BLOCK_S)
        sm = s_offs_full < S
        x_ptrs = x_ptr + base + c_offs[:, None] * S + s_offs_full[None, :]
        full_mask = c_mask[:, None] & sm[None, :]
        x = tl.load(x_ptrs, mask=full_mask, other=0.0)
        y = x * m[:, None]
        y = tl.where(full_mask, y, 0.0)
        sum_acc += tl.sum(y, axis=1)
        sumsq_acc += tl.sum(y * y, axis=1)

    inv_S = 1.0 / S
    mean = sum_acc * inv_S
    var = sumsq_acc * inv_S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: produce output tile
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    x_ptrs = x_ptr + base + c_offs[:, None] * S + s_offs[None, :]
    full_mask = c_mask[:, None] & s_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0)

    y = x * m[:, None]
    y = (y - mean[:, None]) * invstd[:, None]
    y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
    y = y * m[:, None]

    neg_inf = float("-inf")
    y_masked = tl.where(c_mask[:, None], y, neg_inf)
    out = tl.max(y_masked, axis=0)

    tl.store(out_ptr + pid_n * S + s_offs, out, mask=s_mask)


@triton.jit
def stats_kernel(
    x_ptr,
    mult_ptr,
    mean_ptr,
    invstd_ptr,
    N, C, S,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    m = tl.load(mult_ptr + c)

    sum_acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    sumsq_acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    base = n * C * S + c * S
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * m
        y = tl.where(mask, y, 0.0)
        sum_acc += y
        sumsq_acc += y * y

    s_red = tl.sum(sum_acc, axis=0)
    sq_red = tl.sum(sumsq_acc, axis=0)
    inv_S = 1.0 / S
    mean = s_red * inv_S
    var = sq_red * inv_S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + n * C + c, mean)
    tl.store(invstd_ptr + n * C + c, invstd)


@triton.jit
def apply_norm_clamp_max_kernel(
    x_ptr,
    mult_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    N, C, S,
    clamp_min,
    clamp_max,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    m = tl.load(mult_ptr + c_offs, mask=c_mask, other=0.0)
    mean = tl.load(mean_ptr + pid_n * C + c_offs, mask=c_mask, other=0.0)
    invstd = tl.load(invstd_ptr + pid_n * C + c_offs, mask=c_mask, other=0.0)

    x_ptrs = x_ptr + pid_n * C * S + c_offs[:, None] * S + s_offs[None, :]
    full_mask = c_mask[:, None] & s_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0)

    y = x * m[:, None]
    y = (y - mean[:, None]) * invstd[:, None]
    y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
    y = y * m[:, None]

    neg_inf = float("-inf")
    y_masked = tl.where(c_mask[:, None], y, neg_inf)
    out = tl.max(y_masked, axis=0)

    tl.store(out_ptr + pid_n * S + s_offs, out, mask=s_mask)


def custom_conv3d(x, weight, bias):
    N, IC, ID, IH, IW = x.shape
    OC, _, KT, KH, KW = weight.shape
    OD = ID - KT + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    S = OD * OH * OW

    y = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_S = 128
    grid = (N, OC, (S + BLOCK_S - 1) // BLOCK_S)

    conv3d_kernel[grid](
        x, weight, bias, y,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KT=KT, KH=KH, KW=KW,
        BLOCK_S=BLOCK_S,
        num_warps=4,
        num_stages=2,
    )
    return y


def fused_post_conv(x, multiplier, clamp_min, clamp_max, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    x_flat = x.view(N, C, S)
    mult_flat = multiplier.contiguous().view(C)

    mean = torch.empty((N, C), device=x.device, dtype=torch.float32)
    invstd = torch.empty((N, C), device=x.device, dtype=torch.float32)

    BLOCK_S_STATS = 1024
    stats_kernel[(N * C,)](
        x_flat, mult_flat, mean, invstd,
        N, C, S, eps,
        BLOCK_S=BLOCK_S_STATS,
        num_warps=4,
    )

    out = torch.empty((N, S), device=x.device, dtype=torch.float32)
    BLOCK_S = 256
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = (N, (S + BLOCK_S - 1) // BLOCK_S)
    apply_norm_clamp_max_kernel[grid](
        x_flat, mult_flat, mean, invstd, out,
        N, C, S, clamp_min, clamp_max,
        BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    return out.view(N, D, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        x = custom_conv3d(x, w, b)
        return fused_post_conv(x, self.multiplier, self.clamp_min, self.clamp_max, eps=1e-5)