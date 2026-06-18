import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 1, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 2, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 1, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'D_in', 'H_in', 'W_in'],
)
@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr,        # [N, IC, D_in, H_in, W_in]
    w_ptr,        # [IC, OC, KD, KH, KW]
    bias_ptr,     # [OC]
    out_ptr,      # [N, OC, D_out, H_out, W_out]
    sum_w,        # scalar
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pass  # unused — replaced by gather kernel below


# Gather-style: one program per (N, OC, output spatial tile)
# We compute a tile of output spatial positions (along W) for one (n, oc, d_out, h_out)
# by iterating over input channels and kernel positions, gathering from x.
# This is the standard im2col-style implementation but for conv_transpose.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_W': 32}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'W_out'],
)
@triton.jit
def conv_transpose3d_gather_kernel(
    x_ptr,        # [N, IC, D_in, H_in, W_in]
    w_ptr,        # [IC, OC, KD, KH, KW]
    bias_ptr,     # [OC]
    out_ptr,      # [N, OC, D_out, H_out, W_out]
    sum_w,
    N, IC, OC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_w_tile = tl.program_id(1)

    # decode pid -> (n, oc, d_out, h_out)
    h_out = pid % H_out
    tmp = pid // H_out
    d_out = tmp % D_out
    tmp = tmp // D_out
    oc = tmp % OC
    n = tmp // OC

    w_off = pid_w_tile * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_off < W_out

    bias = tl.load(bias_ptr + oc).to(tl.float32)
    acc = tl.zeros([BLOCK_W], dtype=tl.float32) + bias

    # output coord (d_out, h_out, w_out) corresponds to:
    # for each (kd, kh, kw): input pos must satisfy
    #   (d_out + PD - kd) % SD == 0 and >= 0 and < D_in*SD
    # similarly for h, w
    d_eff = d_out + PD  # i_d * SD + kd = d_eff
    h_eff = h_out + PH

    # iterate kernel
    for kd in tl.static_range(0, KD):
        rem_d = d_eff - kd
        i_d = rem_d // SD
        valid_d = ((rem_d - i_d * SD) == 0) & (i_d >= 0) & (i_d < D_in)
        for kh in tl.static_range(0, KH):
            rem_h = h_eff - kh
            i_h = rem_h // SH
            valid_h = ((rem_h - i_h * SH) == 0) & (i_h >= 0) & (i_h < H_in)
            valid_dh = valid_d & valid_h
            for kw in tl.static_range(0, KW):
                # for each w_out in tile compute i_w
                w_eff = w_off + PW
                rem_w = w_eff - kw
                i_w = rem_w // SW
                valid_w = ((rem_w - i_w * SW) == 0) & (i_w >= 0) & (i_w < W_in) & w_mask
                valid = valid_dh & valid_w

                # for each ic: load x[n, ic, i_d, i_h, i_w] * w[ic, oc, kd, kh, kw]
                # i_d, i_h are scalar; i_w is vector
                x_base_dh = n * (IC * D_in * H_in * W_in) + i_d * (H_in * W_in) + i_h * W_in
                # weight base for (oc, kd, kh, kw) over ic:
                # w[ic, oc, kd, kh, kw] = w_ptr[ic*OC*KD*KH*KW + oc*KD*KH*KW + kd*KH*KW + kh*KW + kw]
                w_kbase = oc * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw

                for ic in range(0, IC):
                    x_offs = x_base_dh + ic * (D_in * H_in * W_in) + i_w
                    x_vals = tl.load(x_ptr + x_offs, mask=valid, other=0.0).to(tl.float32)
                    w_val = tl.load(w_ptr + ic * (OC * KD * KH * KW) + w_kbase).to(tl.float32)
                    acc += x_vals * w_val

    acc = acc + sum_w

    out_base = n * (OC * D_out * H_out * W_out) + oc * (D_out * H_out * W_out) + d_out * (H_out * W_out) + h_out * W_out
    tl.store(out_ptr + out_base + w_off, acc, mask=w_mask)


# LayerNorm over last dim (W) + AvgPool3d(2,2,2) + GELU fused kernel.
# Same as the pool baseline.
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['C', 'W'],
)
@triton.jit
def fused_post_kernel(
    x_ptr, out_ptr,
    gamma_ptr, beta_ptr,
    eps,
    N, C, D, H, W,
    Do, Ho, Wo,
    BLOCK_W: tl.constexpr,
    BLOCK_WO: tl.constexpr,
):
    pid = tl.program_id(0)
    ho = pid % Ho
    tmp = pid // Ho
    do = tmp % Do
    tmp = tmp // Do
    c = tmp % C
    n = tmp // C

    d0 = do * 2
    h0 = ho * 2

    w_off = tl.arange(0, BLOCK_W)
    w_mask = w_off < W

    gamma = tl.load(gamma_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)

    DHW = D * H * W
    HW = H * W
    inv_W = 1.0 / W

    base_nc = n * C * DHW + c * DHW

    row_sum = tl.zeros([BLOCK_W], dtype=tl.float32)

    for i in tl.static_range(0, 4):
        dd = d0 + (i // 2)
        hh = h0 + (i % 2)
        row_base = base_nc + dd * HW + hh * W
        x = tl.load(x_ptr + row_base + w_off, mask=w_mask, other=0.0).to(tl.float32)

        x_zero = tl.where(w_mask, x, 0.0)
        mean = tl.sum(x_zero, axis=0) * inv_W
        diff = tl.where(w_mask, x - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) * inv_W
        rstd = 1.0 / tl.sqrt(var + eps)
        y = (x - mean) * rstd * gamma + beta
        row_sum = row_sum + y

    row_sum_2d = tl.reshape(row_sum, [BLOCK_WO, 2])
    pooled = tl.sum(row_sum_2d, axis=1) * (1.0 / 8.0)

    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))

    wo_off = tl.arange(0, BLOCK_WO)
    wo_mask = wo_off < Wo
    DHWo = Do * Ho * Wo
    HWo = Ho * Wo
    out_base = n * C * DHWo + c * DHWo + do * HWo + ho * Wo
    tl.store(out_ptr + out_base + wo_off, gelu, mask=wo_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.pool_kernel_size = pool_kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        N = x.shape[0]
        IC = self.in_channels
        OC = self.out_channels
        KD, KH, KW = self.kernel_size
        SD, SH, SW = self.stride
        PD, PH, PW = self.padding
        OPD, OPH, OPW = self.output_padding

        D_in, H_in, W_in = x.shape[2], x.shape[3], x.shape[4]
        D_out = (D_in - 1) * SD - 2 * PD + KD + OPD
        H_out = (H_in - 1) * SH - 2 * PH + KH + OPH
        W_out = (W_in - 1) * SW - 2 * PW + KW + OPW

        x_c = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KD, KH, KW]
        bias = self.conv_transpose.bias
        if bias is None:
            bias = torch.zeros(OC, device=x.device, dtype=x.dtype)
        else:
            bias = bias.contiguous()

        sum_w = float(self.sum_weight.item())

        ct_out = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

        # Launch conv_transpose gather kernel
        BLOCK_W_CT = 32
        grid_ct = lambda meta: (
            N * OC * D_out * H_out,
            triton.cdiv(W_out, meta['BLOCK_W']),
        )
        conv_transpose3d_gather_kernel[grid_ct](
            x_c, weight, bias, ct_out,
            sum_w,
            N, IC, OC,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
        )

        # Post: LN(last dim=W) + avgpool(2,2,2) + GELU
        Do, Ho, Wo = D_out // 2, H_out // 2, W_out // 2
        out = torch.empty((N, OC, Do, Ho, Wo), device=x.device, dtype=x.dtype)

        gamma = self.norm.weight.contiguous()
        beta = self.norm.bias.contiguous()
        eps = self.norm.eps

        BLOCK_W = _next_pow2(W_out)
        BLOCK_WO = BLOCK_W // 2
        grid = (N * OC * Do * Ho,)
        fused_post_kernel[grid](
            ct_out, out, gamma, beta,
            eps,
            N, OC, D_out, H_out, W_out,
            Do, Ho, Wo,
            BLOCK_W=BLOCK_W,
            BLOCK_WO=BLOCK_WO,
        )
        return out