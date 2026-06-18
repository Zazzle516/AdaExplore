import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
    ],
    key=['S', 'IC', 'OC', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv3d_mult_stats_kernel(
    x_ptr, w_ptr, b_ptr, mult_ptr, out_ptr, mean_ptr, invstd_ptr,
    N, IC: tl.constexpr, OC: tl.constexpr,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    S: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    m = tl.load(mult_ptr + pid_c)
    bias = tl.load(b_ptr + pid_c)

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    out_base = pid_n * OC * S + pid_c * S
    in_n_base = pid_n * IC * ID * IH * IW
    w_oc_base = pid_c * IC * KD * KH * KW

    HW_out = OH * OW

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S

        od = s_offs // HW_out
        rem = s_offs % HW_out
        oh = rem // OW
        ow = rem % OW

        acc = tl.zeros((BLOCK_S,), dtype=tl.float32) + bias

        for ic in tl.static_range(0, IC):
            for kd in tl.static_range(0, KD):
                for kh in tl.static_range(0, KH):
                    for kw in tl.static_range(0, KW):
                        id_ = od + kd
                        ih = oh + kh
                        iw = ow + kw
                        in_off = in_n_base + ic * ID * IH * IW + id_ * IH * IW + ih * IW + iw
                        w_off = w_oc_base + ic * KD * KH * KW + kd * KH * KW + kh * KW + kw
                        x_val = tl.load(x_ptr + in_off, mask=mask, other=0.0)
                        w_val = tl.load(w_ptr + w_off)
                        acc += x_val * w_val

        y = acc * m
        tl.store(out_ptr + out_base + s_offs, y, mask=mask)

        y_masked = tl.where(mask, y, 0.0)
        sum_val += tl.sum(y_masked, axis=0)
        sumsq_val += tl.sum(y_masked * y_masked, axis=0)

    mean = sum_val / S
    var = sumsq_val / S - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid_n * OC + pid_c, mean)
    tl.store(invstd_ptr + pid_n * OC + pid_c, inv_std)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
    ],
    key=['S', 'C'],
)
@triton.jit
def fused_norm_clamp_max_kernel(
    x_ptr, mult_ptr, mean_ptr, invstd_ptr, out_ptr,
    N, C: tl.constexpr, S,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    neg_inf = float('-inf')
    acc = tl.full((BLOCK_S,), neg_inf, dtype=tl.float32)

    n_base = pid_n * C * S
    nc_base = pid_n * C

    for c in tl.static_range(0, C):
        m = tl.load(mult_ptr + c)
        mean = tl.load(mean_ptr + nc_base + c)
        invstd = tl.load(invstd_ptr + nc_base + c)
        x = tl.load(x_ptr + n_base + c * S + s_offs, mask=mask, other=0.0)
        y = (x - mean) * invstd
        y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
        y = y * m
        acc = tl.maximum(acc, y)

    tl.store(out_ptr + pid_n * S + s_offs, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        S = OD * OH * OW

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()
        mult_flat = self.multiplier.contiguous().view(-1)

        conv_out = torch.empty((N, OC, S), device=x.device, dtype=torch.float32)
        mean = torch.empty((N, OC), device=x.device, dtype=torch.float32)
        invstd = torch.empty((N, OC), device=x.device, dtype=torch.float32)

        conv3d_mult_stats_kernel[(N, OC)](
            x, weight, bias, mult_flat, conv_out, mean, invstd,
            N, IC, OC,
            ID, IH, IW,
            OD, OH, OW,
            KD, KH, KW,
            S,
            1e-5,
        )

        out = torch.empty((N, OD, OH, OW), device=x.device, dtype=x.dtype)
        out_flat = out.view(N, S)

        grid2 = lambda META: (N, (S + META['BLOCK_S'] - 1) // META['BLOCK_S'])
        fused_norm_clamp_max_kernel[grid2](
            conv_out, mult_flat, mean, invstd, out_flat,
            N, OC, S,
            self.clamp_min, self.clamp_max,
        )

        return out