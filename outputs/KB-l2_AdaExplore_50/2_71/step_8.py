import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32, 'BLOCK_OC': 32, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 16, 'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_OC': 64, 'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 32, 'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
    ],
    key=['N', 'OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_div_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    inv_divisor,
    neg_slope,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_N: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    IC_C: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_n = tl.program_id(2)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    n_mask = n_offs < N
    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    # Use 3D accumulator: [BLOCK_N, BLOCK_OC, BLOCK_HW]
    acc = tl.zeros((BLOCK_N, BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for ic in tl.static_range(0, IC_C):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh  # [BLOCK_HW]
                iw = ow + kw  # [BLOCK_HW]
                # x[n, ic, ih, iw] -> [BLOCK_N, BLOCK_HW]
                x_offs = (n_offs[:, None] * stride_xn
                          + ic * stride_xc
                          + ih[None, :] * stride_xh
                          + iw[None, :] * stride_xw)
                x_m = n_mask[:, None] & hw_mask[None, :] & (ic < IC)
                x_val = tl.load(x_ptr + x_offs, mask=x_m, other=0.0)  # [BN, BHW]

                # w[oc, ic, kh, kw] -> [BLOCK_OC]
                w_offs = oc_offs * stride_wo + ic * stride_wi + kh * stride_wh + kw * stride_ww
                w_m = oc_mask & (ic < IC)
                w_val = tl.load(w_ptr + w_offs, mask=w_m, other=0.0)  # [BOC]

                # outer product accumulate: x[BN, BHW] * w[BOC] -> [BN, BOC, BHW]
                acc += x_val[:, None, :] * w_val[None, :, None]

    # bias [BOC]
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_val[None, :, None]

    # divide and leaky relu
    acc = acc * inv_divisor
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    # store output[n, oc, oh, ow]
    out_offs = (n_offs[:, None, None] * stride_on
                + oc_offs[None, :, None] * stride_oc
                + oh[None, None, :] * stride_oh
                + ow[None, None, :] * stride_ow)
    out_mask = n_mask[:, None, None] & oc_mask[None, :, None] & hw_mask[None, None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = float(divisor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv.weight.contiguous().cuda()
        bias = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = weight.shape[0]
        KH = weight.shape[2]
        KW = weight.shape[3]
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(OH * OW, meta['BLOCK_HW']),
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        conv2d_div_lrelu_kernel[grid](
            x, weight, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            1.0 / self.divisor,
            0.01,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            IC_C=IC,
        )
        return out