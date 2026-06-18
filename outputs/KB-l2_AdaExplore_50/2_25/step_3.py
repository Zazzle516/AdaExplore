import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 32, 'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64, 'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 32, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_OC': 32}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    ow_offs = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offs < OW

    # Initialize min accumulator
    min_val = tl.full((BLOCK_OW,), float('inf'), dtype=tl.float32)

    # Loop over output channels in blocks of BLOCK_OC
    for oc_start in range(0, OC, BLOCK_OC):
        oc_offs = oc_start + tl.arange(0, BLOCK_OC)
        oc_mask = oc_offs < OC

        # load bias
        bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
        # acc shape: [BLOCK_OC, BLOCK_OW]
        acc = bias[:, None] + tl.zeros((BLOCK_OC, BLOCK_OW), dtype=tl.float32)

        # Convolution loop
        for ic in range(0, IC):
            for kh in tl.static_range(0, KH):
                ih = pid_oh + kh
                for kw in tl.static_range(0, KW):
                    iw = ow_offs + kw  # [BLOCK_OW]
                    # input pointer: n, ic, ih, iw
                    x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                    x_mask = ow_mask  # ih is valid by construction since OH = IH - KH + 1
                    x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_OW]
                    # weight pointer: oc, ic, kh, kw
                    w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                    w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                    acc += w_vals[:, None] * x_vals[None, :]

        # Mask invalid channels with +inf
        acc = tl.where(oc_mask[:, None], acc, float('inf'))
        # reduce min over channel block
        block_min = tl.min(acc, axis=0)  # [BLOCK_OW]
        min_val = tl.minimum(min_val, block_min)

    # Apply tanh(tanh(min))
    t1 = (tl.exp(2 * min_val) - 1) / (tl.exp(2 * min_val) + 1)
    t2 = (tl.exp(2 * t1) - 1) / (tl.exp(2 * t1) + 1)

    out_off = pid_n * (OH * OW) + pid_oh * OW + ow_offs
    tl.store(out_ptr + out_off, t2, mask=ow_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        N, IC, IH, IW = x.shape
        OC = w.shape[0]
        KH = w.shape[2]
        KW = w.shape[3]
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (N, OH, triton.cdiv(OW, meta['BLOCK_OW']))
        conv_min_tanh_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
        )
        return out