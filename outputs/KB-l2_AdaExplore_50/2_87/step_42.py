import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 64, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 2, 'BLOCK_OW': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 1, 'BLOCK_OW': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 1, 'BLOCK_OW': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OH': 2, 'BLOCK_OW': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'KH', 'KW', 'OH', 'OW'],
)
@triton.jit
def conv2d_sub_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SUB: tl.constexpr,
    BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr, BLOCK_OC: tl.constexpr,
):
    pid_oh = tl.program_id(0)
    pid_ow = tl.program_id(1)
    pid_bc = tl.program_id(2)

    num_oc_blocks = tl.cdiv(OC, BLOCK_OC)
    pid_b = pid_bc // num_oc_blocks
    pid_oc = pid_bc % num_oc_blocks

    oh_offs = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)  # [BLOCK_OH]
    ow_offs = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]

    oh_mask = oh_offs < OH
    ow_mask = ow_offs < OW
    oc_mask = oc_offs < OC

    # Accumulator: [BLOCK_OH, BLOCK_OW, BLOCK_OC] flatten to 2D for tl.dot-like ops
    acc = tl.zeros((BLOCK_OH * BLOCK_OW, BLOCK_OC), dtype=tl.float32)

    x_batch_base = pid_b * IC * IH * IW
    w_oc_base = oc_offs * (IC * KH * KW)  # [BLOCK_OC]

    spatial_mask_2d = oh_mask[:, None] & ow_mask[None, :]  # [BLOCK_OH, BLOCK_OW]
    spatial_mask = tl.reshape(spatial_mask_2d, (BLOCK_OH * BLOCK_OW,))

    for ic in range(0, IC):
        for kh in range(0, KH):
            ih_offs = oh_offs + kh  # [BLOCK_OH]
            for kw in range(0, KW):
                iw_offs = ow_offs + kw  # [BLOCK_OW]
                # x[pid_b, ic, ih, iw]
                x_addr_2d = x_batch_base + ic * IH * IW + ih_offs[:, None] * IW + iw_offs[None, :]
                x_addr = tl.reshape(x_addr_2d, (BLOCK_OH * BLOCK_OW,))
                x_vals = tl.load(x_ptr + x_addr, mask=spatial_mask, other=0.0)  # [BLOCK_OH*BLOCK_OW]

                w_addr = w_oc_base + ic * KH * KW + kh * KW + kw
                w_vals = tl.load(w_ptr + w_addr, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += x_vals[:, None] * w_vals[None, :]

    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_vals[None, :] - SUB

    # mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = acc * th

    # store: out[b, oc, oh, ow]; layout NCHW
    # out flat addr: pid_b*OC*OH*OW + oc*OH*OW + oh*OW + ow
    spatial_idx_2d = oh_offs[:, None] * OW + ow_offs[None, :]  # [BLOCK_OH, BLOCK_OW]
    spatial_idx = tl.reshape(spatial_idx_2d, (BLOCK_OH * BLOCK_OW,))
    out_addr = pid_b * OC * OH * OW + oc_offs[None, :] * (OH * OW) + spatial_idx[:, None]
    out_mask = spatial_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_addr, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
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

        sub = float(self.subtract_value_1 + self.subtract_value_2)

        grid = lambda meta: (
            triton.cdiv(OH, meta['BLOCK_OH']),
            triton.cdiv(OW, meta['BLOCK_OW']),
            N * triton.cdiv(OC, meta['BLOCK_OC']),
        )

        conv2d_sub_mish_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            sub,
        )
        return out