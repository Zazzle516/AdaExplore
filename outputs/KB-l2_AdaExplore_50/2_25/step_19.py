import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 2, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 2, 'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 2, 'BLOCK_OW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 1, 'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 1, 'BLOCK_OW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC: tl.constexpr, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oh_blk = tl.program_id(1)
    pid_ow_blk = tl.program_id(2)

    oh_offs = pid_oh_blk * BLOCK_OH + tl.arange(0, BLOCK_OH)
    ow_offs = pid_ow_blk * BLOCK_OW + tl.arange(0, BLOCK_OW)
    oh_mask = oh_offs < OH
    ow_mask = ow_offs < OW

    oc_offs = tl.arange(0, OC)

    bias = tl.load(b_ptr + oc_offs)  # [OC]
    # acc: [OC, BLOCK_OH, BLOCK_OW]
    acc = bias[:, None, None] + tl.zeros((OC, BLOCK_OH, BLOCK_OW), dtype=tl.float32)

    x_mask = oh_mask[:, None] & ow_mask[None, :]
    n_base = pid_n * (IC * IH * IW)

    for kh in tl.static_range(0, KH):
        ih = oh_offs + kh
        for kw in tl.static_range(0, KW):
            iw = ow_offs + kw
            row_base = n_base + ih[:, None] * IW + iw[None, :]  # [BLOCK_OH, BLOCK_OW]
            w_kk_base = oc_offs * (IC * KH * KW) + kh * KW + kw  # [OC]
            for ic in tl.static_range(0, IC):
                x_off = row_base + ic * (IH * IW)
                x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_OH, BLOCK_OW]
                w_off = w_kk_base + ic * (KH * KW)
                w_vals = tl.load(w_ptr + w_off)  # [OC]
                acc += w_vals[:, None, None] * x_vals[None, :, :]

    # min over OC
    min_val = tl.min(acc, axis=0)  # [BLOCK_OH, BLOCK_OW]

    # tanh(tanh(x))
    e1 = tl.exp(2.0 * min_val)
    t1 = (e1 - 1.0) / (e1 + 1.0)
    e2 = tl.exp(2.0 * t1)
    t2 = (e2 - 1.0) / (e2 + 1.0)

    out_off = pid_n * (OH * OW) + oh_offs[:, None] * OW + ow_offs[None, :]
    out_mask = oh_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptr + out_off, t2, mask=out_mask)


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