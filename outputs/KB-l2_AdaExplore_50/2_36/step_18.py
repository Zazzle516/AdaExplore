import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 64, 'BLOCK_OH': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 64, 'BLOCK_OH': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 128, 'BLOCK_OH': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 128, 'BLOCK_OH': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 64, 'BLOCK_OH': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_W': 128, 'BLOCK_OH': 8}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW'],
)
@triton.jit
def conv_transpose_min_sum_partial_kernel(
    x_ptr, w_ptr, b_ptr, acc_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_OH: tl.constexpr,
):
    # grid: (N, OH_tiles, OW_tiles)
    pid_n = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_w = tl.program_id(2)

    w_start = pid_w * BLOCK_W
    ow_offs = w_start + tl.arange(0, BLOCK_W)
    ow_mask = ow_offs < OW

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    ic_offs = tl.arange(0, BLOCK_IC)
    ic_mask = ic_offs < IC

    # Load conv bias (OC,)
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # (BLOCK_OC,)

    x_n_base = pid_n * IC * IH * IW

    # Hoist all KH*KW weight tiles out of OH loop. Pre-load into separate variables.
    # Each tile is (BLOCK_OC, BLOCK_IC)
    w_mask = oc_mask[:, None] & ic_mask[None, :]
    w_base = oc_offs[:, None] * (KH * KW) + ic_offs[None, :] * (OC * KH * KW)

    partial = tl.zeros((BLOCK_W,), dtype=tl.float32)

    oh_start = pid_oh * BLOCK_OH
    for oh_local in range(0, BLOCK_OH):
        oh = oh_start + oh_local
        oh_valid = oh < OH

        conv_tile = b_vals[:, None] + tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32)

        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih = ih_num // SH
            ih_ok = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih >= 0) & (ih < IH) & oh_valid

            for kw in tl.static_range(0, KW):
                iw_num = ow_offs + PW - kw
                iw = iw_num // SW
                iw_ok = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw >= 0) & (iw < IW)
                col_valid = iw_ok & ow_mask  # (BLOCK_W,)

                if ih_ok:
                    w_off = w_base + (kh * KW + kw)
                    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                    x_off = (x_n_base
                             + ic_offs[:, None] * (IH * IW)
                             + ih * IW
                             + iw[None, :])
                    x_mask = ic_mask[:, None] & col_valid[None, :]
                    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                    conv_tile += tl.dot(w_tile, x_tile, out_dtype=tl.float32)

        conv_tile = tl.where(oc_mask[:, None], conv_tile, float('inf'))
        min_vec = tl.min(conv_tile, axis=0)
        # Only accumulate if oh is valid
        min_vec = tl.where(oh_valid, min_vec, 0.0)
        partial += min_vec

    # atomic_add into acc[pid_n, ow_offs]
    acc_offs = pid_n * OW + ow_offs
    tl.atomic_add(acc_ptr + acc_offs, partial, mask=ow_mask)


@triton.jit
def gelu_bias_kernel(
    acc_ptr, bias_ptr, out_ptr,
    NUMEL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUMEL
    acc = tl.load(acc_ptr + offs, mask=mask, other=0.0)
    inv_sqrt2 = 0.7071067811865475
    gelu_out = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))
    bias_val = tl.load(bias_ptr)
    result = gelu_out + bias_val
    tl.store(out_ptr + offs, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OP = self.output_padding
        OH = (IH - 1) * SH - 2 * PH + KH + OP
        OW = (IW - 1) * SW - 2 * PW + KW + OP

        weight = self.conv_transpose.weight.contiguous()
        conv_bias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        bias = self.bias.contiguous()

        out = torch.empty((N, 1, 1, OW), device=x.device, dtype=x.dtype)
        acc = torch.zeros((N, OW), device=x.device, dtype=torch.float32)

        # pick BLOCK_OC as next pow2 >= OC
        def next_pow2(n):
            p = 1
            while p < n:
                p *= 2
            return p

        BLOCK_OC = next_pow2(OC)
        BLOCK_IC = next_pow2(IC)

        grid = lambda META: (N, triton.cdiv(OH, META['BLOCK_OH']), triton.cdiv(OW, META['BLOCK_W']))

        conv_transpose_min_sum_partial_kernel[grid](
            x, weight, conv_bias, acc,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW, SH, SW, PH, PW,
            BLOCK_IC=BLOCK_IC,
            BLOCK_OC=BLOCK_OC,
        )

        NUMEL = N * OW
        BLOCK = 256
        gelu_bias_kernel[(triton.cdiv(NUMEL, BLOCK),)](
            acc, bias, out,
            NUMEL,
            BLOCK=BLOCK,
            num_warps=4,
            num_stages=2,
        )

        return out