import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_W': 32}, num_warps=4, num_stages=3),
    ],
    key=['N', 'IC', 'ID', 'IH', 'IW', 'OC'],
)
@triton.jit
def conv3d_kernel(
    x_ptr,           # [N, IC, ID, IH, IW]
    w_ptr,           # [OC, IC, KT, KH, KW]
    b_ptr,           # [OC]
    out_ptr,         # [N, OC, OD, OH, OW]
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KT: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid: (N*OD*OH, OC/BLOCK_OC, ceil(OW/BLOCK_W))
    pid_ndh = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_w = tl.program_id(2)

    n = pid_ndh // (OD * OH)
    rem = pid_ndh % (OD * OH)
    od = rem // OH
    oh = rem % OH

    ow_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    ow_mask = ow_offs < OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Accumulator [BLOCK_OC, BLOCK_W]
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32) + bias[:, None]

    IH_IW = IH * IW
    KH_KW = KH * KW
    KT_KH_KW = KT * KH_KW

    # Loop over IC, KT, KH, KW
    for ic in range(0, IC):
        for kt in tl.static_range(0, KT):
            id_ = od + kt
            x_base_t = ((n * IC + ic) * ID + id_) * IH_IW
            w_base_t = oc_offs * (IC * KT_KH_KW) + ic * KT_KH_KW + kt * KH_KW
            for kh in tl.static_range(0, KH):
                ih = oh + kh
                x_base_h = x_base_t + ih * IW
                w_base_h = w_base_t + kh * KW
                for kw in tl.static_range(0, KW):
                    iw = ow_offs + kw
                    x_off = x_base_h + iw
                    x_vals = tl.load(x_ptr + x_off, mask=ow_mask, other=0.0)  # [BLOCK_W]
                    w_off = w_base_h + kw  # [BLOCK_OC]
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                    acc += w_val[:, None] * x_vals[None, :]

    out_off = ((n * OC + oc_offs[:, None]) * OD + od) * OH * OW + oh * OW + ow_offs[None, :]
    out_mask = oc_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


@triton.jit
def stats_kernel(
    x_ptr,           # [N, C, S]
    mult_ptr,        # [C]
    mean_ptr,        # [N, C]
    invstd_ptr,      # [N, C]
    N, C, S,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    m = tl.load(mult_ptr + c)
    sum_acc = 0.0
    sumsq_acc = 0.0

    base = n * C * S + c * S
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * m
        sum_acc += tl.sum(tl.where(mask, y, 0.0), axis=0)
        sumsq_acc += tl.sum(tl.where(mask, y * y, 0.0), axis=0)

    mean = sum_acc / S
    var = sumsq_acc / S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + n * C + c, mean)
    tl.store(invstd_ptr + n * C + c, invstd)


@triton.jit
def fused_norm_clamp_max_kernel(
    x_ptr,           # [N, C, S]
    mult_ptr,        # [C]
    mean_ptr,        # [N, C]
    invstd_ptr,      # [N, C]
    out_ptr,         # [N, S]
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


def triton_conv3d(x, weight, bias):
    N, IC, ID, IH, IW = x.shape
    OC, _, KT, KH, KW = weight.shape
    OD = ID - KT + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_OC = 4
    if OC % 8 == 0:
        BLOCK_OC = 8

    grid = lambda meta: (N * OD * OH, (OC + BLOCK_OC - 1) // BLOCK_OC, (OW + meta['BLOCK_W'] - 1) // meta['BLOCK_W'])
    conv3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KT=KT, KH=KH, KW=KW,
        BLOCK_OC=BLOCK_OC,
    )
    return out


def fused_post_conv(x, multiplier, clamp_min, clamp_max, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    x_flat = x.contiguous().view(N, C, S)
    mult_flat = multiplier.contiguous().view(C)

    mean = torch.empty((N, C), device=x.device, dtype=torch.float32)
    invstd = torch.empty((N, C), device=x.device, dtype=torch.float32)

    BLOCK_S_STATS = 512
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
    fused_norm_clamp_max_kernel[grid](
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

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        x = triton_conv3d(x, w, b)
        return fused_post_conv(x, self.multiplier, self.clamp_min, self.clamp_max, eps=1e-5)