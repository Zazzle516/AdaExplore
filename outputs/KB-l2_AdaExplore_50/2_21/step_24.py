import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=3),
    ],
    key=['OH', 'OW', 'OC', 'IC_C'],
)
@triton.jit
def conv_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, bias_ptr, scale_ptr,
    out_ptr,
    N, IC, IH, IW, OC, OH, OW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    IC_C: tl.constexpr,
):
    pid_m = tl.program_id(0)  # spatial tile
    pid_n = tl.program_id(1)  # OC tile (full = 1)
    pid_b = tl.program_id(2)  # batch

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channels

    OHW = OH * OW
    oh = offs_m // OW
    ow = offs_m % OW

    m_mask = offs_m < OHW
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop: IC * KH * KW
    for ic in tl.static_range(0, IC_C):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh
                iw = ow + kw
                # load input slice [BLOCK_M]
                x_off = pid_b * IC * IH * IW + ic * IH * IW + ih * IW + iw
                x_vals = tl.load(x_ptr + x_off, mask=m_mask, other=0.0)  # [BLOCK_M]
                # load weight slice [BLOCK_N]
                w_off = offs_n * (IC * KH * KW) + ic * KH * KW + kh * KW + kw
                w_vals = tl.load(w_ptr + w_off, mask=n_mask, other=0.0)  # [BLOCK_N]
                acc += x_vals[:, None] * w_vals[None, :]

    # add conv bias
    cb = tl.load(conv_bias_ptr + offs_n, mask=n_mask, other=0.0)  # [BLOCK_N]
    acc += cb[None, :]

    # add bias, scale, sigmoid
    bv = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    sv = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
    v = (acc + bv[None, :]) * sv[None, :]
    v = tl.sigmoid(v)

    # store as (N, OC, OHW): out[b, oc, m]
    out_off = pid_b * OC * OHW + offs_n[None, :] * OHW + offs_m[:, None]
    mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, v, mask=mask)


@triton.jit
def gn_kernel(
    x_ptr,
    gn_weight_ptr, gn_bias_ptr,
    out_ptr,
    N, C, HW,
    num_groups,
    group_size,
    eps,
    BLOCK_HW: tl.constexpr,
    CPG: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups
    c_start = g * CPG

    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    base = n * C * HW + c_start * HW

    for ci in tl.static_range(0, CPG):
        for hw_start in range(0, HW, BLOCK_HW):
            offs = hw_start + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            ptr = x_ptr + base + ci * HW + offs
            x = tl.load(ptr, mask=mask, other=0.0)
            sum_x += tl.sum(x, axis=0)
            sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / group_size
    var = sum_x2 / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for ci in tl.static_range(0, CPG):
        c = c_start + ci
        gw = tl.load(gn_weight_ptr + c)
        gb = tl.load(gn_bias_ptr + c)
        for hw_start in range(0, HW, BLOCK_HW):
            offs = hw_start + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            ptr = x_ptr + base + ci * HW + offs
            x = tl.load(ptr, mask=mask, other=0.0)
            y = (x - mean) * rstd * gw + gb
            tl.store(out_ptr + base + ci * HW + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        OC = self.out_channels
        OH = IH - KH + 1
        OW = IW - KW + 1
        OHW = OH * OW

        w = self.conv.weight.contiguous()  # (OC, IC, KH, KW)
        cb = self.conv.bias.contiguous()
        bias_flat = self.bias.view(-1).contiguous()
        scale_flat = self.scale.view(-1).contiguous()

        # Output of fused conv: (N, OC, OH, OW)
        fused_out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_N = 32  # full OC
        grid = lambda meta: (triton.cdiv(OHW, meta['BLOCK_M']), triton.cdiv(OC, BLOCK_N), N)

        conv_fused_kernel[grid](
            x, w, cb, bias_flat, scale_flat,
            fused_out,
            N, IC, IH, IW, OC, OH, OW,
            BLOCK_N=BLOCK_N,
            KH=KH, KW=KW, IC_C=IC,
        )

        # GroupNorm
        out = torch.empty_like(fused_out)
        HW = OH * OW
        CPG = OC // self.num_groups
        group_size = CPG * HW
        BLOCK_HW = 2048
        gn_grid = (N * self.num_groups,)
        gn_kernel[gn_grid](
            fused_out,
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            out,
            N, OC, HW,
            self.num_groups,
            group_size,
            self.eps,
            BLOCK_HW=BLOCK_HW,
            CPG=CPG,
            num_warps=8,
            num_stages=3,
        )
        return out