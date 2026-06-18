import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,        # [N, IC, ID, IH, IW]
    w_ptr,        # [OC, IC, KD, KH, KW]
    b_ptr,        # [OC]
    out_ptr,      # [N, OC, OD, OH, OW]
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    OC_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    OS = OD * OH * OW
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    s_mask = s_offs < OS

    od = s_offs // (OH * OW)
    rem = s_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_offs = tl.arange(0, OC_C)  # [OC_C]
    oc_mask = oc_offs < OC

    # accumulator [OC_C, BLOCK_S]
    acc = tl.zeros([OC_C, BLOCK_S], dtype=tl.float32)

    ic_offs = tl.arange(0, IC_C)  # [IC_C]
    ic_mask = ic_offs < IC

    for kd in tl.static_range(0, KD):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                id_ = od + kd
                ih_ = oh + kh
                iw_ = ow + kw
                # input load: shape [IC_C, BLOCK_S]
                x_offsets = (pid_n * IC * ID * IH * IW
                             + ic_offs[:, None] * (ID * IH * IW)
                             + id_[None, :] * (IH * IW)
                             + ih_[None, :] * IW
                             + iw_[None, :])
                x_mask = ic_mask[:, None] & s_mask[None, :]
                x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)  # [IC_C, BLOCK_S]

                # weight load: shape [OC_C, IC_C]
                w_offsets = (oc_offs[:, None] * (IC * KD * KH * KW)
                             + ic_offs[None, :] * (KD * KH * KW)
                             + kd * (KH * KW) + kh * KW + kw)
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)  # [OC_C, IC_C]

                acc += tl.dot(w_vals, x_vals)

    # add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [OC_C]
    acc += bias[:, None]

    # store [OC_C, BLOCK_S] -> out[N, OC, OS]
    out_offsets = (pid_n * OC * OS
                   + oc_offs[:, None] * OS
                   + s_offs[None, :])
    out_mask = oc_mask[:, None] & s_mask[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


@triton.jit
def stats_kernel(
    x_ptr,        # [N, C, S]
    mult_ptr,     # [C]
    mean_ptr,     # [N, C]
    invstd_ptr,   # [N, C]
    N, C, S,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    m = tl.load(mult_ptr + c)
    base = n * C * S + c * S

    sum1 = 0.0
    sum2 = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * m
        y = tl.where(mask, y, 0.0)
        sum1 += tl.sum(y, axis=0)
        sum2 += tl.sum(y * y, axis=0)

    mean = sum1 / S
    var = sum2 / S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + n * C + c, mean)
    tl.store(invstd_ptr + n * C + c, invstd)


@triton.jit
def apply_max_kernel(
    x_ptr,        # [N, C, S]
    mult_ptr,     # [C]
    mean_ptr,     # [N, C]
    invstd_ptr,   # [N, C]
    out_ptr,      # [N, S]
    N, C, S,
    clamp_min,
    clamp_max,
    BLOCK_S: tl.constexpr,
    C_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    NEG_INF = -float('inf')
    max_val = tl.full([BLOCK_S], NEG_INF, dtype=tl.float32)

    for c_i in tl.static_range(0, C_CONST):
        if c_i < C:
            m = tl.load(mult_ptr + c_i)
            mu = tl.load(mean_ptr + pid_n * C + c_i)
            iv = tl.load(invstd_ptr + pid_n * C + c_i)
            x = tl.load(x_ptr + pid_n * C * S + c_i * S + s_offs, mask=s_mask, other=0.0)
            y = x * m
            y = (y - mu) * iv
            y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
            y = y * m
            max_val = tl.maximum(max_val, y)

    tl.store(out_ptr + pid_n * S + s_offs, max_val, mask=s_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super().__init__()
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
        OS = OD * OH * OW

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        conv_out = torch.empty((N, OC, OS), device=x.device, dtype=torch.float32)

        # OC_C >= OC, IC_C >= IC, both power of 2
        OC_C = 1
        while OC_C < OC:
            OC_C *= 2
        OC_C = max(OC_C, 16)
        IC_C = 1
        while IC_C < IC:
            IC_C *= 2
        IC_C = max(IC_C, 16)

        BLOCK_S_CONV = 128
        grid_conv = (N, triton.cdiv(OS, BLOCK_S_CONV))
        conv3d_kernel[grid_conv](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD=KD, KH=KH, KW=KW,
            IC_C=IC_C, OC_C=OC_C,
            BLOCK_S=BLOCK_S_CONV,
            num_warps=4,
        )

        mult_flat = self.multiplier.contiguous().view(-1)

        mean = torch.empty((N, OC), device=x.device, dtype=torch.float32)
        invstd = torch.empty((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_S = 1024
        stats_kernel[(N * OC,)](
            conv_out, mult_flat, mean, invstd,
            N, OC, OS, 1e-5,
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )

        C_CONST = 1
        while C_CONST < OC:
            C_CONST *= 2

        out = torch.empty((N, OS), device=x.device, dtype=torch.float32)
        grid = (N, triton.cdiv(OS, BLOCK_S))
        apply_max_kernel[grid](
            conv_out, mult_flat, mean, invstd, out,
            N, OC, OS,
            self.clamp_min, self.clamp_max,
            BLOCK_S=BLOCK_S,
            C_CONST=C_CONST,
            num_warps=4,
        )

        return out.view(N, OD, OH, OW)