import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
    ],
    key=['OC', 'OH', 'OW', 'IC'],
)
@triton.jit
def conv2d_div_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IH, IW,
    OC, OH, OW,
    inv_div, neg_slope,
    IC: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (OH * OW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    ic_range = tl.arange(0, IC)
    # x is NHWC: [N, IH, IW, IC]
    x_batch_base = pid_n * (IH * IW * IC)
    # w is [OC, KH, KW, IC]
    w_oc_base = oc_offs[:, None] * (KH * KW * IC)

    for kh in tl.static_range(KH):
        ih = oh + kh  # [BLOCK_HW]
        ih_row = ih * IW  # [BLOCK_HW]
        for kw in tl.static_range(KW):
            iw = ow + kw  # [BLOCK_HW]
            # x offset: [BLOCK_HW, IC]
            x_off = x_batch_base + (ih_row + iw)[:, None] * IC + ic_range[None, :]
            x_val = tl.load(x_ptr + x_off, mask=hw_mask[:, None], other=0.0)  # [BLOCK_HW, IC]

            # w offset: [BLOCK_OC, IC]
            w_off = w_oc_base + (kh * KW + kw) * IC + ic_range[None, :]
            w_val = tl.load(w_ptr + w_off, mask=oc_mask[:, None], other=0.0)  # [BLOCK_OC, IC]

            acc += tl.dot(w_val, tl.trans(x_val))

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_val[:, None]
    acc = acc * inv_div
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    # output NHWC: [N, OH, OW, OC] -> but we'll write as NCHW for compatibility
    # Actually write NHWC, then convert outside
    out_off = pid_n * (OH * OW * OC) + hw_offs[:, None] * OC + oc_offs[None, :]
    out_mask = hw_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, tl.trans(acc), mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        # Convert to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w_nhwc = self.conv.weight.permute(0, 2, 3, 1).contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']), triton.cdiv(OH * OW, meta['BLOCK_HW']))

        conv2d_div_lrelu_kernel[grid](
            x_nhwc, w_nhwc, b, out_nhwc,
            N, IH, IW,
            OC, OH, OW,
            1.0 / float(self.divisor), 0.01,
            IC=IC,
            KH=KH,
            KW=KW,
        )
        return out_nhwc.permute(0, 3, 1, 2).contiguous()