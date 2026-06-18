import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 256}, num_warps=8, num_stages=3),
    ],
    key=['OC', 'OH', 'OW', 'IC'],
)
@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    SCALE: tl.constexpr,
    INV_SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_ow = tl.program_id(0)
    pid_row = tl.program_id(1)  # combined: oh * cdiv(OC,BLOCK_OC) + oc_block
    pid_n = tl.program_id(2)

    num_oc_blocks = (OC + BLOCK_OC - 1) // BLOCK_OC
    oh = pid_row // num_oc_blocks
    pid_oc = pid_row % num_oc_blocks

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    ow_offs = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)

    oc_mask = oc_offs < OC
    ow_mask = ow_offs < OW

    # Precompute padded coords
    oh_pad = oh + PAD
    ow_pad = ow_offs + PAD

    acc = tl.zeros((BLOCK_OC, BLOCK_OW), dtype=tl.float32)

    x_batch_off = pid_n * (IC * IH * IW)
    ic_offs = tl.arange(0, BLOCK_IC)

    for kh in tl.static_range(0, KH):
        ih_num = oh_pad - kh
        ih = ih_num // STRIDE
        kh_valid = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)
        if kh_valid:
            for kw in tl.static_range(0, KW):
                iw_num = ow_pad - kw
                iw = iw_num // STRIDE
                valid = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW) & ow_mask

                x_addrs = x_batch_off + ic_offs[:, None] * (IH * IW) + ih * IW + iw[None, :]
                x_tile = tl.load(x_ptr + x_addrs, mask=valid[None, :], other=0.0)

                w_addrs = ic_offs[None, :] * (OC * KH * KW) + oc_offs[:, None] * (KH * KW) + kh * KW + kw
                w_tile = tl.load(w_ptr + w_addrs, mask=oc_mask[:, None], other=0.0)

                acc += tl.dot(w_tile, x_tile, allow_tf32=True)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]

    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * SCALE
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * INV_SCALE

    out_off = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + oh * OW + ow_offs[None, :]
    out_mask = oc_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = (IH - 1) * self.stride - 2 * self.padding + KH + self.output_padding
        OW = (IW - 1) * self.stride - 2 * self.padding + KW + self.output_padding

        fused_bias = (self.conv_transpose.bias + self.bias.view(-1)).contiguous()
        weight = self.conv_transpose.weight.contiguous()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_IC = max(16, triton.next_power_of_2(IC))
        grid = lambda meta: (
            triton.cdiv(OW, meta['BLOCK_OW']),
            OH * triton.cdiv(OC, meta['BLOCK_OC']),
            N,
        )

        conv_transpose2d_kernel[grid](
            x, weight, fused_bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            self.stride, self.padding,
            float(self.scaling_factor),
            float(1.0 / self.scaling_factor),
            BLOCK_IC=BLOCK_IC,
        )
        return out