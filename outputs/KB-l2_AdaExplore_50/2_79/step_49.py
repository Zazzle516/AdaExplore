import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_S': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=3),
    ],
    key=['N', 'IC', 'D', 'H', 'W', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv3d_kernel(
    x_ptr,        # [N, IC, D, H, W]
    w_ptr,        # [OC, IC, KD, KH, KW]
    b_ptr,        # [OC]
    out_ptr,      # [N, OC, OD, OH, OW]
    N, IC,
    D, H, W,
    OD, OH, OW,
    OC: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    S = OD * OH * OW
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    od = s_offs // (OH * OW)
    rem = s_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_offs = tl.arange(0, OC)  # [OC]

    acc = tl.zeros((OC, BLOCK_S), dtype=tl.float32)

    x_base = pid_n * IC * D * H * W

    for kd in tl.static_range(0, KD):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                id_ = od + kd  # [BLOCK_S]
                ih = oh + kh
                iw = ow + kw
                for ic in tl.static_range(0, IC_C):
                    w_off = oc_offs * (IC_C * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w = tl.load(w_ptr + w_off)  # [OC]

                    x_off = x_base + ic * (D * H * W) + id_ * (H * W) + ih * W + iw  # [BLOCK_S]
                    x = tl.load(x_ptr + x_off, mask=s_mask, other=0.0)  # [BLOCK_S]

                    acc += w[:, None] * x[None, :]

    bias = tl.load(b_ptr + oc_offs)  # [OC]
    acc += bias[:, None]

    out_base = pid_n * OC * S
    out_offs = out_base + oc_offs[:, None] * S + s_offs[None, :]
    out_mask = s_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


@triton.jit
def fused_stats_apply_max_kernel(
    x_ptr,        # [N, C, S]
    mult_ptr,     # [C]
    out_ptr,      # [N, S]
    N, S,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
    C: tl.constexpr,
):
    pid_n = tl.program_id(0)

    c_offs = tl.arange(0, C)
    mult = tl.load(mult_ptr + c_offs)  # [C]

    # Pass 1: compute mean and var per channel using one pass with sums.
    sum1 = tl.zeros((C,), dtype=tl.float32)
    sum2 = tl.zeros((C,), dtype=tl.float32)

    base = pid_n * C * S

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        # load [C, BLOCK_S]
        offs = base + c_offs[:, None] * S + s_offs[None, :]
        x = tl.load(x_ptr + offs, mask=s_mask[None, :], other=0.0)
        y = x * mult[:, None]
        y = tl.where(s_mask[None, :], y, 0.0)
        sum1 += tl.sum(y, axis=1)
        sum2 += tl.sum(y * y, axis=1)

    mean = sum1 / S
    var = sum2 / S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: apply norm, clamp, mult, max-over-C
    out_base = pid_n * S
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        offs = base + c_offs[:, None] * S + s_offs[None, :]
        x = tl.load(x_ptr + offs, mask=s_mask[None, :], other=0.0)
        y = x * mult[:, None]
        y = (y - mean[:, None]) * invstd[:, None]
        y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
        y = y * mult[:, None]
        m = tl.max(y, axis=0)  # [BLOCK_S]
        tl.store(out_ptr + out_base + s_offs, m, mask=s_mask)


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
        N, IC, D, H, W = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1
        S = OD * OH * OW

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        conv_out = torch.empty((N, OC, S), device=x.device, dtype=torch.float32)

        grid_conv = lambda meta: (N, triton.cdiv(S, meta['BLOCK_S']))
        conv3d_kernel[grid_conv](
            x, weight, bias, conv_out,
            N, IC,
            D, H, W,
            OD, OH, OW,
            OC=OC,
            KD=KD, KH=KH, KW=KW,
            IC_C=IC,
        )

        mult_flat = self.multiplier.contiguous().view(-1)

        out = torch.empty((N, S), device=x.device, dtype=torch.float32)

        BLOCK_S2 = 1024
        fused_stats_apply_max_kernel[(N,)](
            conv_out, mult_flat, out,
            N, S,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max,
            eps=1e-5,
            BLOCK_S=BLOCK_S2,
            C=OC,
            num_warps=8,
        )

        return out.view(N, OD, OH, OW)