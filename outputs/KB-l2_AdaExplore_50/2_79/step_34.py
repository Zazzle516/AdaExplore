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
    IC_CONST: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OC_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_s = tl.program_id(1)  # spatial tile

    s_offs = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)
    S = OD * OH * OW
    mask_s = s_offs < S

    # decompose s into (od, oh, ow)
    od = s_offs // (OH * OW)
    rem = s_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    # accumulator [BLOCK_N, OC_CONST]
    acc = tl.zeros((BLOCK_N, OC_CONST), dtype=tl.float32)

    oc_idx = tl.arange(0, OC_CONST)

    for ic in tl.static_range(0, IC_CONST):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    id_ = od + kd
                    ih = oh + kh
                    iw = ow + kw
                    x_off = pid_n * IC * ID * IH * IW + ic * ID * IH * IW + id_ * IH * IW + ih * IW + iw
                    x_val = tl.load(x_ptr + x_off, mask=mask_s, other=0.0)  # [BLOCK_N]
                    # weights [OC, IC, KD, KH, KW]
                    w_off = oc_idx * (IC_CONST * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)  # [OC]
                    acc += x_val[:, None] * w_val[None, :]

    # add bias
    b = tl.load(b_ptr + oc_idx)  # [OC]
    acc += b[None, :]

    # store: out shape [N, OC, S], contiguous in S
    out_base = pid_n * OC_CONST * S
    for c in tl.static_range(0, OC_CONST):
        tl.store(out_ptr + out_base + c * S + s_offs, acc[:, c], mask=mask_s)


@triton.jit
def mean_invstd_kernel(
    x_ptr, mult_ptr, mean_ptr, invstd_ptr,
    N, C, S,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    base = pid_n * C * S + pid_c * S
    m = tl.load(mult_ptr + pid_c)

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * m
        y = tl.where(mask, y, 0.0)
        sum_val += tl.sum(y, axis=0)
        sumsq_val += tl.sum(y * y, axis=0)

    mean = sum_val / S
    var = sumsq_val / S - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid_n * C + pid_c, mean)
    tl.store(invstd_ptr + pid_n * C + pid_c, inv_std)


@triton.jit
def fused_norm_clamp_max_kernel(
    x_ptr, mult_ptr, mean_ptr, invstd_ptr, out_ptr,
    N, C, S,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK_S: tl.constexpr,
    C_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    neg_inf = float('-inf')
    acc = tl.full((BLOCK_S,), neg_inf, dtype=tl.float32)

    n_base = pid_n * C * S
    nc_base = pid_n * C

    for c in tl.static_range(0, C_CONST):
        m = tl.load(mult_ptr + c)
        mean = tl.load(mean_ptr + nc_base + c)
        invstd = tl.load(invstd_ptr + nc_base + c)
        x = tl.load(x_ptr + n_base + c * S + s_offs, mask=mask, other=0.0)
        y = x * m
        y = (y - mean) * invstd
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
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        S = OD * OH * OW

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_N = 128
        grid_conv = (N, (S + BLOCK_N - 1) // BLOCK_N)
        conv3d_kernel[grid_conv](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            IC,
            BLOCK_N,
            OC,
            num_warps=4,
            num_stages=2,
        )

        x_flat = conv_out.view(N, OC, S)
        mult_flat = self.multiplier.contiguous().view(-1)

        mean = torch.empty((N, OC), device=x.device, dtype=torch.float32)
        invstd = torch.empty((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_S1 = 1024
        mean_invstd_kernel[(N, OC)](
            x_flat, mult_flat, mean, invstd,
            N, OC, S,
            1e-5,
            BLOCK_S=BLOCK_S1,
            num_warps=4,
        )

        out = torch.empty((N, OD, OH, OW), device=x.device, dtype=x.dtype)
        out_flat = out.view(N, S)

        BLOCK_S2 = 512
        grid2 = (N, (S + BLOCK_S2 - 1) // BLOCK_S2)
        fused_norm_clamp_max_kernel[grid2](
            x_flat, mult_flat, mean, invstd, out_flat,
            N, OC, S,
            self.clamp_min, self.clamp_max,
            BLOCK_S=BLOCK_S2,
            C_CONST=OC,
            num_warps=4,
            num_stages=2,
        )

        return out