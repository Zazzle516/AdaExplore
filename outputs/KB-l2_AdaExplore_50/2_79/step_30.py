import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv3d_kernel(
    x_ptr,       # [N, IC, ID, IH, IW]
    w_ptr,       # [OC, IC, KD, KH, KW]
    b_ptr,       # [OC]
    mult_ptr,    # [OC]
    out_ptr,     # [N, OC, OD, OH, OW]
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    S = OD * OH * OW
    s_mask = s_offs < S

    # decompose s -> (od, oh, ow)
    od = s_offs // (OH * OW)
    rem = s_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros([BLOCK_OC, BLOCK_S], dtype=tl.float32)

    # Loop over IC, KD, KH, KW
    for ic in range(0, IC):
        for kd in tl.static_range(0, KD):
            id_ = od + kd
            for kh in tl.static_range(0, KH):
                ih = oh + kh
                for kw in tl.static_range(0, KW):
                    iw = ow + kw
                    x_offs = (
                        pid_n * IC * ID * IH * IW
                        + ic * ID * IH * IW
                        + id_ * IH * IW
                        + ih * IW
                        + iw
                    )
                    x_val = tl.load(x_ptr + x_offs, mask=s_mask, other=0.0)  # [BLOCK_S]
                    w_offs = (
                        oc_offs * (IC * KD * KH * KW)
                        + ic * (KD * KH * KW)
                        + kd * (KH * KW)
                        + kh * KW
                        + kw
                    )
                    w_val = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                    acc += w_val[:, None] * x_val[None, :]

    # Add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + bias[:, None]
    # Multiply by multiplier (first multiplication)
    mult = tl.load(mult_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc * mult[:, None]

    # store result [N, OC, S]
    out_offs = pid_n * OC * S + oc_offs[:, None] * S + s_offs[None, :]
    mask2d = oc_mask[:, None] & s_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=mask2d)


@triton.jit
def _fused_stats_norm_clamp_max_kernel(
    x_ptr,       # [N, C, S]  (post conv * mult)
    mult_ptr,    # [C]
    out_ptr,     # [N, S]
    N, C: tl.constexpr, S,
    clamp_min, clamp_max, eps,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    c_offs = tl.arange(0, C)

    # Pass 1: compute per-channel mean and var over full S for this n.
    # Use accumulators of shape [C].
    sum_x = tl.zeros([C], dtype=tl.float32)
    sum_x2 = tl.zeros([C], dtype=tl.float32)

    NUM_BLOCKS_S = (S + BLOCK_S - 1) // BLOCK_S
    for sb in range(0, NUM_BLOCKS_S):
        s_offs_i = sb * BLOCK_S + tl.arange(0, BLOCK_S)
        sm = s_offs_i < S
        offs = pid_n * C * S + c_offs[:, None] * S + s_offs_i[None, :]
        mask2d = sm[None, :]
        x = tl.load(x_ptr + offs, mask=mask2d, other=0.0)
        x = tl.where(mask2d, x, 0.0)
        sum_x += tl.sum(x, axis=1)
        sum_x2 += tl.sum(x * x, axis=1)

    inv_S = 1.0 / S
    mean = sum_x * inv_S
    var = sum_x2 * inv_S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    mult = tl.load(mult_ptr + c_offs)

    # Pass 2: compute output for this s-block
    offs = pid_n * C * S + c_offs[:, None] * S + s_offs[None, :]
    mask2d = s_mask[None, :]
    x = tl.load(x_ptr + offs, mask=mask2d, other=0.0)
    normed = (x - mean[:, None]) * invstd[:, None]
    clamped = tl.minimum(tl.maximum(normed, clamp_min), clamp_max)
    val = clamped * mult[:, None]
    max_val = tl.max(val, axis=0)
    tl.store(out_ptr + pid_n * S + s_offs, max_val, mask=s_mask)


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
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        mult = self.multiplier.view(-1).contiguous()

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        S = OD * OH * OW

        conv_out = torch.empty((N, OC, S), device=x.device, dtype=x.dtype)

        BLOCK_S = 64
        BLOCK_OC = 16
        grid_conv = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(S, BLOCK_S))
        _conv3d_kernel[grid_conv](
            x, w, b, mult, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD=KD, KH=KH, KW=KW,
            BLOCK_S=BLOCK_S, BLOCK_OC=BLOCK_OC,
            num_warps=4, num_stages=2,
        )

        out = torch.empty((N, S), device=x.device, dtype=x.dtype)

        BLOCK_S2 = 256
        grid2 = (N, triton.cdiv(S, BLOCK_S2))
        _fused_stats_norm_clamp_max_kernel[grid2](
            conv_out, mult, out,
            N, OC, S,
            self.clamp_min, self.clamp_max, self.eps,
            BLOCK_S=BLOCK_S2,
            num_warps=4, num_stages=2,
        )

        return out.view(N, OD, OH, OW)