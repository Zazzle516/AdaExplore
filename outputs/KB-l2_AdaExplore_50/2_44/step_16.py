import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'IC_TILE': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'IC_TILE': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64, 'IC_TILE': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128, 'IC_TILE': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256, 'IC_TILE': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64, 'IC_TILE': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'IC_TILE': 64}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv_transpose_fused_mean_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    multiplier_over_hw,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    IC_TILE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    ic_offs = tl.arange(0, IC_TILE)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_num = oh + PAD_H - kh
            iw_num = ow + PAD_W - kw
            ih_num_c = tl.where(ih_num >= 0, ih_num, 0)
            iw_num_c = tl.where(iw_num >= 0, iw_num, 0)
            ih = ih_num_c // STRIDE_H
            iw = iw_num_c // STRIDE_W
            valid = (ih_num >= 0) & (iw_num >= 0) & \
                    ((ih_num % STRIDE_H) == 0) & ((iw_num % STRIDE_W) == 0) & \
                    (ih < IH) & (iw < IW) & hw_mask

            for ic_start in range(0, IC, IC_TILE):
                ic_idx = ic_start + ic_offs
                ic_mask = ic_idx < IC

                x_idx = pid_n * (IC * IH * IW) + ic_idx[:, None] * (IH * IW) + (ih * IW + iw)[None, :]
                x_mask = ic_mask[:, None] & valid[None, :]
                x_tile = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

                w_idx = ic_idx[:, None] * (OC * KH * KW) + oc_offs[None, :] * (KH * KW) + (kh * KW + kw)
                w_mask = ic_mask[:, None] & oc_mask[None, :]
                w_tile = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

                acc += tl.dot(tl.trans(w_tile), x_tile)

    # add bias on valid positions only (so partial sum across HW tiles is correct)
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]
    # zero-out invalid hw positions before reduction
    acc = tl.where(hw_mask[None, :], acc, 0.0)

    # reduce along HW within this tile
    partial = tl.sum(acc, axis=1)  # [BLOCK_OC]
    partial = partial * multiplier_over_hw

    # atomic add into output [N, OC]
    out_idx = pid_n * OC + oc_offs
    tl.atomic_add(out_ptr + out_idx, partial, mask=oc_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()
        bias = self.conv_transpose.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OPH = OPW = self.output_padding

        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        HW = OH * OW
        out = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']), triton.cdiv(HW, meta['BLOCK_HW']))

        conv_transpose_fused_mean_kernel[grid](
            x, weight, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            SH, SW,
            PH, PW,
            float(self.multiplier) / float(HW),
        )

        return out.view(N, OC, 1, 1).to(x.dtype)