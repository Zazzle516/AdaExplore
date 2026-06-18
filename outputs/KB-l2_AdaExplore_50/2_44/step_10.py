import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 32, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC'],
)
@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # program ids: (n, oc_tile, hw_tile)
    n = tl.program_id(0)
    oc_tile = tl.program_id(1)
    hw_tile = tl.program_id(2)

    oc_offs = oc_tile * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    hw_offs = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    oh = hw_offs // OW
    ow = hw_offs % OW
    valid_hw = hw_offs < (OH * OW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    x_n_base = x_ptr + n * (IC * IH * IW)

    for kh in tl.static_range(0, KH):
        h_num = oh + PH - kh
        ih = h_num // SH
        h_valid = (h_num >= 0) & ((h_num % SH) == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            w_num = ow + PW - kw
            iw = w_num // SW
            w_valid = (w_num >= 0) & ((w_num % SW) == 0) & (iw >= 0) & (iw < IW)
            hw_valid = h_valid & w_valid & valid_hw

            # safe indices
            ih_safe = tl.where(hw_valid, ih, 0)
            iw_safe = tl.where(hw_valid, iw, 0)

            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                # x[n, ic, ih, iw] -> (BLOCK_IC, BLOCK_HW)
                x_ptrs = x_n_base + ic_offs[:, None] * (IH * IW) + ih_safe[None, :] * IW + iw_safe[None, :]
                x_mask = ic_mask[:, None] & hw_valid[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                # w[ic, oc, kh, kw] -> (BLOCK_IC, BLOCK_OC)
                w_ptrs = w_ptr + ic_offs[:, None] * (OC * KH * KW) + oc_offs[None, :] * (KH * KW) + kh * KW + kw
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                # acc[oc, hw] += sum_ic w[ic, oc] * x[ic, hw]
                acc += tl.dot(tl.trans(w_vals), x_vals)

    # Add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]

    # Store
    out_mask = oc_mask[:, None] & valid_hw[None, :]
    out_ptrs = out_ptr + n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + hw_offs[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def conv_transpose2d_triton(x, weight, bias, stride, padding, output_padding):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    SH, SW = stride, stride
    PH, PW = padding, padding
    OPH, OPW = output_padding, output_padding

    OH = (IH - 1) * SH - 2 * PH + KH + OPH
    OW = (IW - 1) * SW - 2 * PW + KW + OPW

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(OH * OW, META['BLOCK_HW']))

    conv_transpose_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
    )
    return out


@triton.jit
def mean_scale_kernel(
    inp_ptr, out_ptr,
    N, C, HW,
    scale,
    BLOCK: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * (C * HW) + c * HW

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, HW, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HW
        vals = tl.load(inp_ptr + base + offs, mask=mask, other=0.0)
        acc += vals

    total = tl.sum(acc, axis=0)
    result = total * scale / HW
    tl.store(out_ptr + pid, result)


def mean_scale_triton(x, scale):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty((N, C), device=x.device, dtype=x.dtype)
    grid = (N * C,)
    BLOCK = 1024
    mean_scale_kernel[grid](x, out, N, C, HW, scale, BLOCK=BLOCK, num_warps=4)
    return out.view(N, C, 1, 1)


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
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()
        y = conv_transpose2d_triton(x, weight, bias, self.stride, self.padding, self.output_padding)
        # Fused scale + mean over H,W
        out = mean_scale_triton(y, self.multiplier)
        return out