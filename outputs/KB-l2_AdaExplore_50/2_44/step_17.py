import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Input-stationary scatter-add: each program loads a tile of x for one (n, ic_tile, hw_tile),
# multiplies by weight slice for each (kh, kw) to produce contributions, and accumulates
# per-(n, oc) sum directly (fusing global mean) via atomic_add. This avoids materializing
# the full (N, OC, OH, OW) output.

def _get_configs():
    return [
        triton.Config({'BLOCK_OC': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_IC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
    ]


@triton.autotune(configs=_get_configs(), key=['IC', 'OC', 'IH', 'IW'])
@triton.jit
def conv_transpose_fused_mean_kernel(
    x_ptr, w_ptr, sum_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # Each program owns a unique (n, oc_tile); no atomics.
    n = tl.program_id(0)
    oc_tile = tl.program_id(1)

    oc_offs = oc_tile * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    in_hw = IH * IW

    # For each (kh, kw), we need s_ic[ic] = sum over valid (ih, iw) of x[n, ic, ih, iw].
    # Then oc_sum[oc] += sum_ic w[ic, oc, kh, kw] * s_ic[ic].

    oc_sum = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    hw_range = tl.arange(0, BLOCK_HW)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                # accumulate s_ic over all hw chunks
                s_ic = tl.zeros((BLOCK_IC,), dtype=tl.float32)
                for hw_start in range(0, in_hw, BLOCK_HW):
                    hw_offs = hw_start + hw_range
                    valid_hw = hw_offs < in_hw
                    ih = hw_offs // IW
                    iw = hw_offs % IW
                    oh = ih * SH - PH + kh
                    ow = iw * SW - PW + kw
                    oh_valid = (oh >= 0) & (oh < OH)
                    ow_valid = (ow >= 0) & (ow < OW)
                    pos_valid = oh_valid & ow_valid & valid_hw  # (BLOCK_HW,)

                    x_ptrs = x_ptr + n * (IC * in_hw) + ic_offs[:, None] * in_hw + hw_offs[None, :]
                    x_mask = ic_mask[:, None] & pos_valid[None, :]
                    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # (BLOCK_IC, BLOCK_HW)
                    s_ic += tl.sum(x_vals, axis=1)

                # Load weight slice (BLOCK_IC, BLOCK_OC) for this (kh, kw)
                w_ptrs = w_ptr + ic_offs[:, None] * (OC * KH * KW) + oc_offs[None, :] * (KH * KW) + kh * KW + kw
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # (BLOCK_IC, BLOCK_OC)

                oc_sum += tl.sum(w_vals * s_ic[:, None], axis=0)

    out_ptrs = sum_ptr + n * OC + oc_offs
    tl.store(out_ptrs, oc_sum, mask=oc_mask)


@triton.jit
def finalize_kernel(
    sum_ptr, bias_ptr, out_ptr,
    N, OC,
    inv_hw, scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < (N * OC)
    oc = offs % OC
    s = tl.load(sum_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + oc, mask=mask, other=0.0)
    result = (s * inv_hw + b) * scale
    tl.store(out_ptr + offs, result, mask=mask)


def conv_transpose2d_fused_mean(x, weight, bias, stride, padding, output_padding, multiplier):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    SH, SW = stride, stride
    PH, PW = padding, padding
    OPH, OPW = output_padding, output_padding

    OH = (IH - 1) * SH - 2 * PH + KH + OPH
    OW = (IW - 1) * SW - 2 * PW + KW + OPW

    sum_buf = torch.empty((N, OC), device=x.device, dtype=torch.float32)
    out = torch.empty((N, OC), device=x.device, dtype=x.dtype)

    inv_hw = 1.0 / (OH * OW)

    # Choose BLOCK_HW as a power-of-two covering at least one row efficiently
    BLOCK_HW = 256

    grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']))

    conv_transpose_fused_mean_kernel[grid](
        x, weight, sum_buf,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        BLOCK_HW=BLOCK_HW,
    )

    BLOCK_F = 128
    total = N * OC
    grid_f = (triton.cdiv(total, BLOCK_F),)
    finalize_kernel[grid_f](
        sum_buf, bias, out,
        N, OC,
        inv_hw, multiplier,
        BLOCK=BLOCK_F,
        num_warps=2,
    )

    return out.view(N, OC, 1, 1)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()
        out = conv_transpose2d_fused_mean(
            x, weight, bias, self.stride, self.padding, self.output_padding, self.multiplier
        )
        return out