import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=3),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_relu_hswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs_oc = tl.arange(0, OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oh = offs_hw // OW
    ow = offs_hw % OW

    mask_hw = offs_hw < (OH * OW)

    acc = tl.zeros((OC, BLOCK_HW), dtype=tl.float32)

    x_base = pid_n * (IC * IH * IW)
    IHW = IH * IW
    KHW = KH * KW
    ICKHKW = IC * KHW

    for ic in tl.static_range(0, IC):
        x_ic_base = x_base + ic * IHW
        w_ic_base = ic * KHW
        for kh in tl.static_range(0, KH):
            ih = oh + kh
            for kw in tl.static_range(0, KW):
                iw = ow + kw
                x_off = x_ic_base + ih * IW + iw
                x_vals = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)

                w_off = offs_oc * ICKHKW + w_ic_base + kh * KW + kw
                w_vals = tl.load(w_ptr + w_off)

                acc += w_vals[:, None] * x_vals[None, :]

    b_vals = tl.load(b_ptr + offs_oc)
    acc = acc + b_vals[:, None]

    acc = tl.maximum(acc, 0.0)
    hs = tl.minimum((acc + 3.0) * (1.0 / 6.0), 1.0)
    acc = acc * hs

    out_off = (pid_n * OC * OH * OW) + offs_oc[:, None] * (OH * OW) + offs_hw[None, :]
    mask_out = mask_hw[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask_out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = w.shape[0]
        KH = w.shape[2]
        KW = w.shape[3]
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (N, triton.cdiv(OH * OW, meta['BLOCK_HW']))

        conv_relu_hswish_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
        )
        return out