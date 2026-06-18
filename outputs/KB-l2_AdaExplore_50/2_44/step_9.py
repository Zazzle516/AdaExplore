import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _get_conv_configs():
    configs = [
        triton.Config({'BLOCK_IC': 32, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_IC': 32, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_IC': 32, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_IC': 64, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_IC': 32, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_IC': 16, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
    ]
    return configs


@triton.autotune(configs=_get_conv_configs(), key=['OC', 'OH', 'OW', 'IC'])
@triton.jit
def conv_transpose_partial_sum_kernel(
    x_ptr, w_ptr, partial_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    NUM_HW_TILES,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # program ids: (n, oc_tile, hw_tile)
    n = tl.program_id(0)
    oc_tile = tl.program_id(1)
    hw_tile = tl.program_id(2)

    oc_offs = oc_tile * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    hw_offsets = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    oh = hw_offsets // OW
    ow = hw_offsets % OW
    valid_hw = hw_offsets < (OH * OW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        h_num = oh + PH - kh
        ih = h_num // SH
        h_valid = (h_num >= 0) & ((h_num % SH) == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            w_num = ow + PW - kw
            iw = w_num // SW
            w_valid = (w_num >= 0) & ((w_num % SW) == 0) & (iw >= 0) & (iw < IW)
            hw_valid = h_valid & w_valid & valid_hw

            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                x_ptrs = x_ptr + n * (IC * IH * IW) + ic_offs[:, None] * (IH * IW) + ih[None, :] * IW + iw[None, :]
                x_mask = ic_mask[:, None] & hw_valid[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                w_ptrs = w_ptr + ic_offs[:, None] * (OC * KH * KW) + oc_offs[None, :] * (KH * KW) + kh * KW + kw
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                acc += tl.dot(tl.trans(w_vals), x_vals)

    acc = tl.where(valid_hw[None, :], acc, 0.0)
    partial = tl.sum(acc, axis=1)  # (BLOCK_OC,)

    # Write to partial[n, oc, hw_tile]
    out_ptrs = partial_ptr + n * (OC * NUM_HW_TILES) + oc_offs * NUM_HW_TILES + hw_tile
    tl.store(out_ptrs, partial, mask=oc_mask)


@triton.jit
def reduce_mean_kernel(
    partial_ptr, bias_ptr, out_ptr,
    N, OC, NUM_HW_TILES,
    inv_hw, scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    base = n * (OC * NUM_HW_TILES) + oc * NUM_HW_TILES

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, NUM_HW_TILES, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < NUM_HW_TILES
        vals = tl.load(partial_ptr + base + offs, mask=mask, other=0.0)
        acc += vals

    total = tl.sum(acc, axis=0)
    bias_val = tl.load(bias_ptr + oc)
    result = (total * inv_hw + bias_val) * scale
    tl.store(out_ptr + pid, result)


def conv_transpose2d_mean_triton(x, weight, bias, stride, padding, output_padding, multiplier):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    SH, SW = stride, stride
    PH, PW = padding, padding
    OPH, OPW = output_padding, output_padding

    OH = (IH - 1) * SH - 2 * PH + KH + OPH
    OW = (IW - 1) * SW - 2 * PW + KW + OPW

    BLOCK_HW = 128
    NUM_HW_TILES = triton.cdiv(OH * OW, BLOCK_HW)

    partial = torch.empty((N, OC, NUM_HW_TILES), device=x.device, dtype=torch.float32)
    out = torch.empty((N, OC), device=x.device, dtype=x.dtype)

    inv_hw = 1.0 / (OH * OW)

    grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']), NUM_HW_TILES)

    conv_transpose_partial_sum_kernel[grid](
        x, weight, partial,
        N, IC, IH, IW,
        OC, OH, OW,
        NUM_HW_TILES,
        KH, KW,
        SH, SW,
        PH, PW,
        BLOCK_HW=BLOCK_HW,
    )

    # Reduction kernel
    BLOCK_R = triton.next_power_of_2(max(NUM_HW_TILES, 16))
    BLOCK_R = min(BLOCK_R, 1024)
    reduce_mean_kernel[(N * OC,)](
        partial, bias, out,
        N, OC, NUM_HW_TILES,
        inv_hw, multiplier,
        BLOCK=BLOCK_R,
        num_warps=4,
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
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()
        out = conv_transpose2d_mean_triton(
            x, weight, bias, self.stride, self.padding, self.output_padding, self.multiplier
        )
        return out