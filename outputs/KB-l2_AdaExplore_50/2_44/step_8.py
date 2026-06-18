import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _get_conv_configs():
    configs = []
    for bhw in [64, 128, 256]:
        for bic in [16, 32, 64]:
            for boc in [16, 32, 64]:
                for nw in [4, 8]:
                    for ns in [2, 3]:
                        configs.append(triton.Config(
                            {'BLOCK_HW': bhw, 'BLOCK_IC': bic, 'BLOCK_OC': boc},
                            num_warps=nw, num_stages=ns))
    return configs


@triton.autotune(configs=_get_conv_configs(), key=['OC', 'OH', 'OW', 'IC'])
@triton.jit
def conv_transpose_fused_mean_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    inv_hw, scale,
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

    # acc: (BLOCK_OC, BLOCK_HW)
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

                # x[n, ic, ih, iw]: (BLOCK_IC, BLOCK_HW)
                x_ptrs = x_ptr + n * (IC * IH * IW) + ic_offs[:, None] * (IH * IW) + ih[None, :] * IW + iw[None, :]
                x_mask = ic_mask[:, None] & hw_valid[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                # w[ic, oc, kh, kw]: (BLOCK_IC, BLOCK_OC)
                w_ptrs = w_ptr + ic_offs[:, None] * (OC * KH * KW) + oc_offs[None, :] * (KH * KW) + kh * KW + kw
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                # GEMM: (BLOCK_OC, BLOCK_IC) @ (BLOCK_IC, BLOCK_HW)
                acc += tl.dot(tl.trans(w_vals), x_vals)

    # Reduce over HW within this tile (sum of acc * scale / HW + bias contribution handled later)
    # acc shape: (BLOCK_OC, BLOCK_HW). Mask invalid HW positions.
    acc = tl.where(valid_hw[None, :], acc, 0.0)
    partial = tl.sum(acc, axis=1)  # (BLOCK_OC,)
    partial = partial * (scale * inv_hw)

    # Atomic add into out[n, oc]
    out_ptrs = out_ptr + n * OC + oc_offs
    tl.atomic_add(out_ptrs, partial, mask=oc_mask)


def conv_transpose2d_mean_triton(x, weight, bias, stride, padding, output_padding, multiplier):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    SH, SW = stride, stride
    PH, PW = padding, padding
    OPH, OPW = output_padding, output_padding

    OH = (IH - 1) * SH - 2 * PH + KH + OPH
    OW = (IW - 1) * SW - 2 * PW + KW + OPW

    # Initialize output with bias contribution: bias * multiplier (since mean of constant = constant)
    out = (bias.float() * multiplier).unsqueeze(0).expand(N, OC).contiguous()

    inv_hw = 1.0 / (OH * OW)

    grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']), triton.cdiv(OH * OW, meta['BLOCK_HW']))

    conv_transpose_fused_mean_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        inv_hw, multiplier,
        KH, KW,
        SH, SW,
        PH, PW,
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