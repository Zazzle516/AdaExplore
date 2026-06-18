import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_div_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    inv_div, neg_slope,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    IC_C: tl.constexpr,
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

    for ic in range(0, IC_C):
        ic_valid = ic < IC
        for kh in tl.static_range(KH):
            ih = oh + kh
            for kw in tl.static_range(KW):
                iw = ow + kw
                # x: [N, IC, IH, IW]
                x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                x_mask = hw_mask & ic_valid
                x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_HW]

                # w: [OC, IC, KH, KW]
                w_off = oc_offs[:, None] * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_mask = oc_mask[:, None] & ic_valid
                w_val = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_OC, 1]

                acc += w_val * x_val[None, :]

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_val[:, None]
    acc = acc * inv_div
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    out_off = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + hw_offs[None, :]
    out_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_HW = 128
        IC_C = IC  # constexpr unroll over input channels

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_HW))

        conv2d_div_lrelu_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            1.0 / float(self.divisor), 0.01,
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            IC_C=IC_C,
            num_warps=4,
            num_stages=2,
        )
        return out