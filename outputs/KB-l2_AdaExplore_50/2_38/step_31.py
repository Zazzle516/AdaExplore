import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _clamp_softmax_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    S,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    C = tl.num_programs(1)
    row_start = (b * C + c) * S

    scale_val = tl.load(scale_ptr + c)

    # pass 1: compute max
    max_val = -float('inf')
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
        v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
        v = tl.where(mask, v, -float('inf'))
        m = tl.max(v, axis=0)
        max_val = tl.maximum(max_val, m)

    # pass 2: sum exp
    sum_val = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
        e = tl.exp(v - max_val)
        e = tl.where(mask, e, 0.0)
        sum_val += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_val

    # pass 3: write output
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
        e = tl.exp(v - max_val) * inv_sum * scale_val
        tl.store(out_ptr + row_start + idx, e, mask=mask)


def fused_clamp_softmax_scale(x, scale, clamp_min, clamp_max):
    B, C, D, H, W = x.shape
    S = D * H * W
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    scale_flat = scale.contiguous().view(-1)

    if S <= 4096:
        BLOCK = triton.next_power_of_2(S)
        nw = 4
    elif S <= 16384:
        BLOCK = 2048
        nw = 8
    else:
        BLOCK = 2048
        nw = 8

    grid = (B, C)
    _clamp_softmax_scale_kernel[grid](
        x_c, scale_flat, out,
        S,
        float(clamp_min), float(clamp_max),
        BLOCK=BLOCK,
        num_warps=nw,
        num_stages=2,
    )
    return out


@triton.jit
def _convtranspose3d_pooled_kernel(
    pooled_ptr,         # (B, IC, Dp, Hp, Wp)
    weight_ptr,         # (IC, OC, KD, KH, KW)
    bias_ptr,           # (OC,)
    out_ptr,            # (B, OC, Do, Ho, Wo)
    B, IC, OC,
    Dp, Hp, Wp,
    Do, Ho, Wo,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # grid: (B*OC, Do*Ho, ceil(Wo/BLOCK_W))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    b = pid0 // OC
    oc = pid0 % OC

    od = pid1 // Ho
    oh = pid1 % Ho

    ow_start = pid2 * BLOCK_W
    ows = ow_start + tl.arange(0, BLOCK_W)
    ow_mask = ows < Wo

    # ConvTranspose: output[b, oc, od, oh, ow] =
    #   sum over ic, kd, kh, kw of input[b, ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
    # where (id*STRIDE - PAD + kd) = od, etc.
    # So: id = (od + PAD - kd) / STRIDE, only valid when divisible & in range.

    bias = tl.load(bias_ptr + oc)
    acc = tl.zeros([BLOCK_W], dtype=tl.float32) + bias

    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_q = id_num // STRIDE
        id_r = id_num - id_q * STRIDE
        d_ok = (id_r == 0) & (id_q >= 0) & (id_q < Dp)

        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_q = ih_num // STRIDE
            ih_r = ih_num - ih_q * STRIDE
            h_ok = (ih_r == 0) & (ih_q >= 0) & (ih_q < Hp)

            for kw in tl.static_range(0, KW):
                iw_num = ows + PAD - kw
                iw_q = iw_num // STRIDE
                iw_r = iw_num - iw_q * STRIDE
                w_ok = (iw_r == 0) & (iw_q >= 0) & (iw_q < Wp) & ow_mask

                valid = d_ok & h_ok & w_ok

                # base pointer for input pixel iw across all ic
                # input idx: ((b*IC + ic)*Dp + id_q)*Hp + ih_q)*Wp + iw_q
                in_spatial = id_q * Hp * Wp + ih_q * Wp + iw_q  # [BLOCK_W]
                in_base = b * IC * Dp * Hp * Wp
                # weight idx: ((ic*OC + oc)*KD + kd)*KH + kh)*KW + kw
                w_kd_kh_kw = kd * KH * KW + kh * KW + kw

                # iterate over ic
                for ic in range(0, IC):
                    in_off = in_base + ic * Dp * Hp * Wp + in_spatial
                    x = tl.load(pooled_ptr + in_off, mask=valid, other=0.0)
                    w_off = ic * OC * KD * KH * KW + oc * KD * KH * KW + w_kd_kh_kw
                    w = tl.load(weight_ptr + w_off)
                    acc += x * w

    out_off = ((b * OC + oc) * Do + od) * Ho * Wo + oh * Wo + ows
    tl.store(out_ptr + out_off, acc, mask=ow_mask)


def conv_transpose3d_triton(pooled, weight, bias, stride, pad, output_padding):
    B, IC, Dp, Hp, Wp = pooled.shape
    _IC, OC, KD, KH, KW = weight.shape
    assert _IC == IC

    Do = (Dp - 1) * stride - 2 * pad + KD + output_padding
    Ho = (Hp - 1) * stride - 2 * pad + KH + output_padding
    Wo = (Wp - 1) * stride - 2 * pad + KW + output_padding

    out = torch.empty((B, OC, Do, Ho, Wo), device=pooled.device, dtype=pooled.dtype)

    BLOCK_W = 32
    grid = (B * OC, Do * Ho, (Wo + BLOCK_W - 1) // BLOCK_W)
    _convtranspose3d_pooled_kernel[grid](
        pooled, weight, bias, out,
        B, IC, OC,
        Dp, Hp, Wp,
        Do, Ho, Wo,
        KD=KD, KH=KH, KW=KW,
        STRIDE=stride, PAD=pad,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        # Use torch's ConvTranspose3d (highly optimized) - custom triton may be slower
        x = self.conv_transpose(x)
        x = fused_clamp_softmax_scale(x, self.scale, self.clamp_min, self.clamp_max)
        return x