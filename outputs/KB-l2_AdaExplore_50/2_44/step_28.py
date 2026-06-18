import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_mean_kernel_v3(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [IC, OC, KH, KW]
    bias_ptr,        # [OC]
    out_ptr,         # [N, OC]
    N, IC, OC, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    multiplier,
    BLOCK_HW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N, OC // BLOCK_OC)
    n = tl.program_id(0)
    oc_block = tl.program_id(1)

    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    IHW = IH * IW
    KHW = KH * KW

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Iterate over input positions
    for hw_start in range(0, IHW, BLOCK_HW):
        hw_offs = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offs < IHW
        ih = hw_offs // IW
        iw = hw_offs % IW

        # For each kernel position, build partial[BLOCK_HW, BLOCK_OC] via tl.dot
        # over IC, then mask invalid output positions and sum over HW.
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                oh = ih * STRIDE_H - PAD_H + kh
                ow = iw * STRIDE_W - PAD_W + kw
                valid = (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW) & hw_mask

                partial = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

                for ic_start in range(0, IC, BLOCK_IC):
                    ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                    ic_mask = ic_offs < IC

                    # x[n, ic, hw]: [BLOCK_HW, BLOCK_IC]
                    x_off = (n * IC * IHW
                             + ic_offs[None, :] * IHW
                             + hw_offs[:, None])
                    x_m = hw_mask[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptr + x_off, mask=x_m, other=0.0)

                    # w[ic, oc, kh, kw]: [BLOCK_IC, BLOCK_OC]
                    w_off = (ic_offs[:, None] * (OC * KHW)
                             + oc_offs[None, :] * KHW
                             + kh * KW + kw)
                    w_m = ic_mask[:, None] & oc_mask[None, :]
                    w_vals = tl.load(w_ptr + w_off, mask=w_m, other=0.0)

                    partial += tl.dot(x_vals, w_vals)

                partial = tl.where(valid[:, None], partial, 0.0)
                acc += tl.sum(partial, axis=0)

    bias = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    OHW = OH * OW
    total = (acc + bias * OHW) * (multiplier / OHW)
    tl.store(out_ptr + n * OC + oc_offs, total, mask=oc_mask)


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
        w = self.conv_transpose.weight.contiguous().cuda()
        b = self.conv_transpose.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding

        OH = (IH - 1) * SH - 2 * PH + KH + self.output_padding
        OW = (IW - 1) * SW - 2 * PW + KW + self.output_padding

        out = torch.empty((N, OC, 1, 1), device=x.device, dtype=torch.float32)

        BLOCK_HW = 128
        BLOCK_IC = 64
        BLOCK_OC = 32

        assert OC % BLOCK_OC == 0, f"OC={OC} must be divisible by BLOCK_OC={BLOCK_OC}"

        grid = (N, OC // BLOCK_OC)
        conv_transpose_mean_kernel_v3[grid](
            x, w, b, out,
            N, IC, OC, IH, IW, OH, OW,
            KH, KW,
            SH, SW, PH, PW,
            float(self.multiplier),
            BLOCK_HW=BLOCK_HW,
            BLOCK_IC=BLOCK_IC,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=3,
        )

        return out