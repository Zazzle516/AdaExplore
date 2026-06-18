import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 64, 'BLOCK_OH': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_OH': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_OH': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 256, 'BLOCK_OH': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 64, 'BLOCK_OH': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_OH': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 64, 'BLOCK_OH': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OW': 64, 'BLOCK_OH': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 128, 'BLOCK_OH': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 32, 'BLOCK_OH': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 32, 'BLOCK_OH': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64, 'BLOCK_OH': 8}, num_warps=8, num_stages=2),
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
    BLOCK_OH: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    ow_offs = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offs < OW

    oh_base = pid_oh * BLOCK_OH
    oh_offs = oh_base + tl.arange(0, BLOCK_OH)
    oh_mask = oh_offs < OH

    oc_offs = tl.arange(0, OC)  # full OC=64

    # Load bias once
    bias = tl.load(b_ptr + oc_offs)  # [OC]
    # Accumulator shape: [OC, BLOCK_OH, BLOCK_OW] -> flatten to [OC, BLOCK_OH*BLOCK_OW]
    BLOCK_OHW: tl.constexpr = BLOCK_OH * BLOCK_OW
    acc = bias[:, None] + tl.zeros((OC, BLOCK_OHW), dtype=tl.float32)

    # row/col within tile
    row_id = tl.arange(0, BLOCK_OH)  # [BLOCK_OH]
    col_id = tl.arange(0, BLOCK_OW)  # [BLOCK_OW]

    # Conv loop
    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            # ih per output row
            ih_vec = oh_base + row_id + kh  # [BLOCK_OH]
            for kw in tl.static_range(0, KW):
                iw_vec = ow_offs + kw  # [BLOCK_OW]
                # gather x: [BLOCK_OH, BLOCK_OW]
                x_off = (pid_n * (IC * IH * IW) + ic * (IH * IW)
                         + ih_vec[:, None] * IW + iw_vec[None, :])
                x_m = oh_mask[:, None] & ow_mask[None, :]
                x_vals = tl.load(x_ptr + x_off, mask=x_m, other=0.0)
                x_flat = tl.reshape(x_vals, (BLOCK_OHW,))

                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_vals = tl.load(w_ptr + w_off)  # [OC]
                acc += w_vals[:, None] * x_flat[None, :]

    # Reduce min across OC
    min_val = tl.min(acc, axis=0)  # [BLOCK_OHW]

    # tanh(tanh(x)) using exp form
    e1 = tl.exp(2.0 * min_val)
    t1 = (e1 - 1.0) / (e1 + 1.0)
    e2 = tl.exp(2.0 * t1)
    t2 = (e2 - 1.0) / (e2 + 1.0)
    t2_2d = tl.reshape(t2, (BLOCK_OH, BLOCK_OW))

    out_off = (pid_n * (OH * OW) + oh_offs[:, None] * OW + ow_offs[None, :])
    out_m = oh_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptr + out_off, t2_2d, mask=out_m)


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

        grid = lambda meta: (N, triton.cdiv(OH, meta['BLOCK_OH']), triton.cdiv(OW, meta['BLOCK_OW']))
        conv_min_tanh_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
        )
        return out