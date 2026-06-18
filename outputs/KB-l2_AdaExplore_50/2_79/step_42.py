import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SP': 384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 384}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SP': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SP': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SP': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 1024}, num_warps=16, num_stages=2),
    ],
    key=['IC', 'OC', 'OD', 'OH', 'OW', 'KD', 'KH', 'KW'],
)
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
    IHIW = IH * IW
    OHOW = OH * OW
    KHKW = KH * KW
    KDHW = KD * KH * KW
    ICKDHW = IC * KD * KH * KW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < OS

    oc_offs = tl.arange(0, OC)

    od = sp_offs // OHOW
    rem = sp_offs % OHOW
    oh = rem // OW
    ow = rem % OW

    bias = tl.load(b_ptr + oc_offs)
    acc = tl.zeros([OC, BLOCK_SP], dtype=tl.float32) + bias[:, None]

    x_base = pid_n * IC * ID * IHIW

    for ic in tl.static_range(0, IC):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw
                    in_idx = x_base + ic * (ID * IHIW) + id_ * IHIW + ih_ * IW + iw_
                    x_val = tl.load(x_ptr + in_idx, mask=sp_mask, other=0.0)
                    w_idx = oc_offs * ICKDHW + ic * KDHW + kd * KHKW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_idx)
                    acc += w_val[:, None] * x_val[None, :]

    out_idx = pid_n * OC * OS + oc_offs[:, None] * OS + sp_offs[None, :]
    mask = sp_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=16, num_stages=2),
    ],
    key=['C', 'S'],
)
@triton.jit
def fused_norm_max_kernel(
    x_ptr,        # [N, C, S]
    mult_ptr,     # [C]
    out_ptr,      # [N, S]
    N, C: tl.constexpr, S: tl.constexpr,
    eps: tl.constexpr,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)

    c_offs = tl.arange(0, C)
    mult = tl.load(mult_ptr + c_offs)  # [C]

    base = pid_n * C * S

    # Pass 1: compute per-channel mean and var
    sum_c = tl.zeros([C], dtype=tl.float32)
    sumsq_c = tl.zeros([C], dtype=tl.float32)

    NSTEPS: tl.constexpr = (S + BLOCK_S - 1) // BLOCK_S
    for sb in tl.static_range(0, NSTEPS):
        s_idx = sb * BLOCK_S + tl.arange(0, BLOCK_S)
        sm = s_idx < S
        ptrs = x_ptr + base + c_offs[:, None] * S + s_idx[None, :]
        v = tl.load(ptrs, mask=sm[None, :], other=0.0)
        v = v * mult[:, None]
        v = tl.where(sm[None, :], v, 0.0)
        sum_c += tl.sum(v, axis=1)
        sumsq_c += tl.sum(v * v, axis=1)

    mean = sum_c / S
    var = sumsq_c / S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize, clamp, *mult, reduce max over C
    for sb in tl.static_range(0, NSTEPS):
        s_idx = sb * BLOCK_S + tl.arange(0, BLOCK_S)
        sm = s_idx < S
        ptrs = x_ptr + base + c_offs[:, None] * S + s_idx[None, :]
        v = tl.load(ptrs, mask=sm[None, :], other=0.0)
        v = v * mult[:, None]
        v = (v - mean[:, None]) * invstd[:, None]
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        v = v * mult[:, None]
        out = tl.max(v, axis=0)
        tl.store(out_ptr + pid_n * S + s_idx, out, mask=sm)


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

        grid_conv = lambda meta: (N, (OS + meta['BLOCK_SP'] - 1) // meta['BLOCK_SP'])
        conv3d_kernel[grid_conv](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
        )

        mult_flat = self.multiplier.view(-1).contiguous()
        x_flat = conv_out.view(N, OC, OS)

        out = torch.empty((N, OS), device=x.device, dtype=x.dtype)
        fused_norm_max_kernel[(N,)](
            x_flat, mult_flat, out,
            N, OC, OS,
            1e-5,
            self.clamp_min, self.clamp_max,
        )

        return out.view(N, OD, OH, OW)