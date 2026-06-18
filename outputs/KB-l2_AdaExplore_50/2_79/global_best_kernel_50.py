import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    OS = OD * OH * OW
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < OS

    oc_offs = tl.arange(0, OC)

    od = sp_offs // (OH * OW)
    rem = sp_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros([OC, BLOCK_SP], dtype=tl.float32)

    x_base = pid_n * IC * ID * IH * IW

    for ic in tl.static_range(0, IC):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw
                    in_idx = x_base + ic * (ID * IH * IW) + id_ * (IH * IW) + ih_ * IW + iw_
                    x_val = tl.load(x_ptr + in_idx, mask=sp_mask, other=0.0)
                    w_idx = oc_offs * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_idx)
                    acc += w_val[:, None] * x_val[None, :]

    bias = tl.load(b_ptr + oc_offs)
    acc += bias[:, None]

    out_idx = pid_n * OC * OS + oc_offs[:, None] * OS + sp_offs[None, :]
    mask = sp_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=mask)


@triton.jit
def stats_kernel(
    x_ptr,        # [N, C, S]
    mult_ptr,     # [C]
    mean_ptr,     # [N, C]
    invstd_ptr,   # [N, C]
    N, C: tl.constexpr, S: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    m = tl.load(mult_ptr + c)

    base = n * C * S + c * S
    sum_v = tl.zeros([BLOCK_S], dtype=tl.float32)
    sumsq_v = tl.zeros([BLOCK_S], dtype=tl.float32)

    for sb in range(0, tl.cdiv(S, BLOCK_S)):
        s_idx = sb * BLOCK_S + tl.arange(0, BLOCK_S)
        sm = s_idx < S
        v = tl.load(x_ptr + base + s_idx, mask=sm, other=0.0)
        v = v * m
        v = tl.where(sm, v, 0.0)
        sum_v += v
        sumsq_v += v * v

    s_sum = tl.sum(sum_v, axis=0)
    s_sumsq = tl.sum(sumsq_v, axis=0)
    mean = s_sum / S
    var = s_sumsq / S - mean * mean
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
    N, C: tl.constexpr, S,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    c_offs = tl.arange(0, C)

    mult = tl.load(mult_ptr + c_offs)   # [C]
    mean = tl.load(mean_ptr + pid_n * C + c_offs)
    invstd = tl.load(invstd_ptr + pid_n * C + c_offs)

    # build effective per-channel transform: out = clamp((x*m - mean)*invstd) * m
    ptrs = x_ptr + pid_n * C * S + c_offs[:, None] * S + s_offs[None, :]
    vals = tl.load(ptrs, mask=s_mask[None, :], other=0.0)
    vals = vals * mult[:, None]
    vals = (vals - mean[:, None]) * invstd[:, None]
    vals = tl.minimum(tl.maximum(vals, clamp_min), clamp_max)
    vals = vals * mult[:, None]

    out = tl.max(vals, axis=0)
    tl.store(out_ptr + pid_n * S + s_offs, out, mask=s_mask)


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
        x = x.contiguous().cuda()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        OS = OD * OH * OW

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_SP = 128
        grid_conv = (N, (OS + BLOCK_SP - 1) // BLOCK_SP)
        conv3d_kernel[grid_conv](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            BLOCK_SP=BLOCK_SP,
            num_warps=4, num_stages=2,
        )

        mult_flat = self.multiplier.view(-1).contiguous()
        mean_buf = torch.empty((N, OC), device=x.device, dtype=torch.float32)
        invstd_buf = torch.empty((N, OC), device=x.device, dtype=torch.float32)

        x_flat = conv_out.view(N, OC, OS)

        BLOCK_S_STATS = 1024
        stats_kernel[(N * OC,)](
            x_flat, mult_flat, mean_buf, invstd_buf,
            N, OC, OS,
            1e-5,
            BLOCK_S=BLOCK_S_STATS,
            num_warps=4, num_stages=2,
        )

        out = torch.empty((N, OS), device=x.device, dtype=x.dtype)
        BLOCK_S_APPLY = 256
        grid_apply = (N, (OS + BLOCK_S_APPLY - 1) // BLOCK_S_APPLY)
        apply_max_kernel[grid_apply](
            x_flat, mult_flat, mean_buf, invstd_buf, out,
            N, OC, OS,
            self.clamp_min, self.clamp_max,
            BLOCK_S=BLOCK_S_APPLY,
            num_warps=4, num_stages=2,
        )

        return out.view(N, OD, OH, OW)