import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv3d_kernel(
    x_ptr,          # [N, IC, ID, IH, IW]
    w_ptr,          # [OC, IC, KD, KH, KW]
    b_ptr,          # [OC]
    out_ptr,        # [N, OC, OD, OH, OW]
    N, IC,
    ID, IH, IW,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OC: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # one program per (n, s_block) - covers all OC channels
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < (OD * OH * OW)

    od = s_offs // (OH * OW)
    rem = s_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    # accumulator [OC, BLOCK_S]
    acc = tl.zeros([OC, BLOCK_S], dtype=tl.float32)

    # K = IC*KD*KH*KW. For this problem 3*3*3*3 = 81.
    oc_offs = tl.arange(0, OC)

    for ic in range(0, IC):
        for kd in range(0, KD):
            id_pos = od + kd  # padding=0
            for kh in range(0, KH):
                ih_pos = oh + kh
                for kw in range(0, KW):
                    iw_pos = ow + kw
                    # input offset
                    x_off = ((pid_n * IC + ic) * ID + id_pos) * IH * IW + ih_pos * IW + iw_pos
                    x_val = tl.load(x_ptr + x_off, mask=s_mask, other=0.0)  # [BLOCK_S]
                    # weight offset: [OC, IC, KD, KH, KW]
                    w_off = oc_offs * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)  # [OC]
                    acc += w_val[:, None] * x_val[None, :]

    # bias
    b = tl.load(b_ptr + oc_offs)  # [OC]
    acc += b[:, None]

    # store: out shape [N, OC, S]
    S = OD * OH * OW
    out_off = pid_n * OC * S + oc_offs[:, None] * S + s_offs[None, :]
    tl.store(out_ptr + out_off, acc, mask=s_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
    ],
    key=['S', 'C'],
)
@triton.jit
def _fused_norm_clamp_max_kernel(
    x_ptr,        # [N, C, S]
    mult_ptr,     # [C]
    out_ptr,      # [N, S]
    S,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    eps: tl.constexpr,
    C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)

    base = pid_n * C * S

    c_offs = tl.arange(0, C)
    mult = tl.load(mult_ptr + c_offs)  # [C]

    # Pass 1: compute mean and var of (x * m) per channel
    sum_x = tl.zeros([C], dtype=tl.float32)
    sum_x2 = tl.zeros([C], dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        offs = base + c_offs[:, None] * S + s_offs[None, :]
        x_raw = tl.load(x_ptr + offs, mask=s_mask[None, :], other=0.0).to(tl.float32)
        x = x_raw * mult[:, None]
        sum_x += tl.sum(x, axis=1)
        sum_x2 += tl.sum(x * x, axis=1)

    inv_S = 1.0 / S
    mean = sum_x * inv_S
    var = sum_x2 * inv_S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize, clamp, multiply, reduce max over C
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        offs = base + c_offs[:, None] * S + s_offs[None, :]
        x_raw = tl.load(x_ptr + offs, mask=s_mask[None, :], other=0.0).to(tl.float32)
        x = x_raw * mult[:, None]
        normed = (x - mean[:, None]) * invstd[:, None]
        clamped = tl.minimum(tl.maximum(normed, clamp_min), clamp_max)
        val = clamped * mult[:, None]
        out_val = tl.max(val, axis=0)
        tl.store(out_ptr + pid_n * S + s_offs, out_val, mask=s_mask)


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
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        OC = self.out_channels
        S = OD * OH * OW

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        conv_out = torch.empty((N, OC, S), device=x.device, dtype=x.dtype)

        # Pick BLOCK_S for conv kernel: spatial S = 14*30*30 = 12600
        BLOCK_S_CONV = 252  # divides nicely; or use power of 2
        # Use 256 for power of 2 (close enough)
        BLOCK_S_CONV = 256
        grid_conv = (N, triton.cdiv(S, BLOCK_S_CONV))
        _conv3d_kernel[grid_conv](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OD, OH, OW,
            KD, KH, KW,
            OC,
            BLOCK_S=BLOCK_S_CONV,
        )

        mult = self.multiplier.view(-1).contiguous()
        out = torch.empty((N, S), device=x.device, dtype=x.dtype)
        grid = (N,)
        _fused_norm_clamp_max_kernel[grid](
            conv_out, mult, out,
            S,
            self.clamp_min, self.clamp_max,
            self.eps,
            C=OC,
        )

        return out.view(N, OD, OH, OW)