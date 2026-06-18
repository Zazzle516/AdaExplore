import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 256, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW, SH, SW, PH, PW,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # program ids: (n, oc_block, sp_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    x_base = pid_n * (IC * IH * IW)
    KHW = KH * KW
    OCKHW = OC * KHW
    IHW = IH * IW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Loop order: kh, kw, ic — spatial validity is computed once per (kh,kw)
    for kh in range(0, KH):
        ih_num = oh + PH - kh
        ih = ih_num // SH
        ih_valid = ((ih_num % SH) == 0) & (ih >= 0) & (ih < IH)
        for kw in range(0, KW):
            iw_num = ow + PW - kw
            iw = iw_num // SW
            iw_valid = ((iw_num % SW) == 0) & (iw >= 0) & (iw < IW)
            valid = ih_valid & iw_valid & sp_mask  # [BLOCK_SP]

            x_sp_off = ih * IW + iw  # [BLOCK_SP]
            w_kpos = kh * KW + kw    # scalar

            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                # x: [BLOCK_IC, BLOCK_SP]
                x_off = x_base + ic_offs[:, None] * IHW + x_sp_off[None, :]
                x_mask = ic_mask[:, None] & valid[None, :]
                x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                # w: [BLOCK_OC, BLOCK_IC]; weight layout (IC, OC, KH, KW)
                w_off = ic_offs[None, :] * OCKHW + oc_offs[:, None] * KHW + w_kpos
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                acc += tl.dot(w_vals, x_vals, allow_tf32=True)

    # add bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_vals[:, None]

    # store: output shape (N, OC, OH, OW)
    out_off = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


@triton.jit
def min_sum_gelu_bias_kernel(
    inp_ptr,   # (N, OC, OH, OW)
    bias_ptr,  # scalar (1,1,1)
    out_ptr,   # (N, 1, 1, OW)
    N, OC, OH, OW,
    BLOCK_OC: tl.constexpr,
    BLOCK_OH: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_w = tl.program_id(1)

    # for this (n, ow), compute sum over h of min over c of inp[n,c,h,ow]
    oc_range = tl.arange(0, BLOCK_OC)
    oh_range = tl.arange(0, BLOCK_OH)

    # We'll iterate over h tiles
    total = tl.zeros((1,), dtype=tl.float32)
    sum_acc = 0.0

    for h_start in range(0, OH, BLOCK_OH):
        oh_offs = h_start + oh_range  # [BLOCK_OH]
        h_mask = oh_offs < OH

        # Compute min over channels for each h
        # Load tile [BLOCK_OC, BLOCK_OH]
        min_vals = tl.full((BLOCK_OH,), float('inf'), dtype=tl.float32)
        for c_start in range(0, OC, BLOCK_OC):
            oc_offs = c_start + oc_range
            c_mask = oc_offs < OC
            # offset: n*OC*OH*OW + c*OH*OW + h*OW + ow
            offs = (pid_n * OC * OH * OW
                    + oc_offs[:, None] * (OH * OW)
                    + oh_offs[None, :] * OW
                    + pid_w)
            mask = c_mask[:, None] & h_mask[None, :]
            vals = tl.load(inp_ptr + offs, mask=mask, other=float('inf'))
            # min along channel dim
            tile_min = tl.min(vals, axis=0)  # [BLOCK_OH]
            min_vals = tl.minimum(min_vals, tile_min)

        # mask invalid h (set to 0 for sum)
        min_vals = tl.where(h_mask, min_vals, 0.0)
        sum_acc += tl.sum(min_vals, axis=0)

    # GELU
    x = sum_acc
    gelu = 0.5 * x * (1.0 + tl.erf(x / 1.4142135623730951))
    # add bias (scalar)
    b = tl.load(bias_ptr)
    res = gelu + b

    out_off = pid_n * OW + pid_w
    tl.store(out_ptr + out_off, res)


def conv_transpose2d_triton(x, weight, bias_param, stride, padding, output_padding):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w
    SH, SW = stride, stride
    PH, PW = padding, padding
    OH = (IH - 1) * SH - 2 * PH + KH + output_padding
    OW = (IW - 1) * SW - 2 * PW + KW + output_padding

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

    grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(OH * OW, META['BLOCK_SP']))
    conv_transpose_kernel[grid](
        x, weight, bias_param, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW, SH, SW, PH, PW,
    )
    return out


def min_sum_gelu_bias_triton(inp, bias):
    N, OC, OH, OW = inp.shape
    out = torch.empty((N, 1, 1, OW), device=inp.device, dtype=torch.float32)
    BLOCK_OC = 128
    BLOCK_OH = 16
    # ensure BLOCK_OC >= OC ideally; tile if not
    grid = (N, OW)
    min_sum_gelu_bias_kernel[grid](
        inp, bias, out,
        N, OC, OH, OW,
        BLOCK_OC=BLOCK_OC, BLOCK_OH=BLOCK_OH,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv_transpose.weight.contiguous().cuda()
        b = self.conv_transpose.bias.contiguous().cuda()
        # weight shape: (IC, OC, KH, KW)
        conv_out = conv_transpose2d_triton(x, w, b, self.stride, self.padding, self.output_padding)
        # bias is (1,1,1) - effectively scalar broadcast over the whole tensor
        # After min, sum, gelu, the result shape is (N, 1, 1, OW). bias broadcasts.
        bias_scalar = self.bias.contiguous().view(-1)[0:1].cuda()
        out = min_sum_gelu_bias_triton(conv_out, bias_scalar)
        return out