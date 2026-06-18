import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OW': 128}, num_warps=8, num_stages=3),
    ],
    key=['IC', 'OC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC: tl.constexpr, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    ow_offs = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offs < OW

    oc_offs = tl.arange(0, OC)

    # load bias: shape [OC]
    bias = tl.load(b_ptr + oc_offs)
    # acc shape: [OC, BLOCK_OW]
    acc = bias[:, None] + tl.zeros((OC, BLOCK_OW), dtype=tl.float32)

    # Convolution loop: load x once per (ic, kh, kw), then w once and accumulate across full OC
    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            ih = pid_oh + kh
            for kw in tl.static_range(0, KW):
                iw = ow_offs + kw  # [BLOCK_OW]
                x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                x_vals = tl.load(x_ptr + x_off, mask=ow_mask, other=0.0)  # [BLOCK_OW]
                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_vals = tl.load(w_ptr + w_off)  # [OC]
                acc += w_vals[:, None] * x_vals[None, :]

    # reduce min over channel dim
    min_val = tl.min(acc, axis=0)  # [BLOCK_OW]

    # Apply tanh(tanh(min))
    e1 = tl.exp(2 * min_val)
    t1 = (e1 - 1) / (e1 + 1)
    e2 = tl.exp(2 * t1)
    t2 = (e2 - 1) / (e2 + 1)

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