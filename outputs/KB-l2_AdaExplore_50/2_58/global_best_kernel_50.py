import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 64, 'BLOCK_H': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 64, 'BLOCK_H': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 64, 'BLOCK_H': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 64, 'BLOCK_H': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 64, 'BLOCK_H': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_W': 64, 'BLOCK_H': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 128, 'BLOCK_H': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 128, 'BLOCK_H': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_W': 128, 'BLOCK_H': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 128, 'BLOCK_H': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_W': 128, 'BLOCK_H': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 256, 'BLOCK_H': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 256, 'BLOCK_H': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_W': 32, 'BLOCK_H': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 32, 'BLOCK_H': 4}, num_warps=4, num_stages=2),
    ],
    key=['OD', 'OH', 'OW', 'IC', 'OC'],
)
@triton.jit
def fused_convtranspose3d_lse_hswish_sub_clamp_kernel(
    x_ptr,        # input: [N, IC, ID, IH, IW]
    w_ptr,        # weight: [IC, OC, KD, KH, KW]
    cb_ptr,       # conv bias: [OC]
    bias_ptr,     # scalar bias
    out_ptr,      # output: [N, 1, OD, OH, OW]
    N, OD, OH, OW,
    ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    IC: tl.constexpr, OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_W: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # grid: (N, OD, cdiv(OH, BLOCK_H) * cdiv(OW, BLOCK_W))
    pid_n = tl.program_id(0)
    od = tl.program_id(1)
    pid_hw = tl.program_id(2)
    num_w_tiles = (OW + BLOCK_W - 1) // BLOCK_W
    pid_h = pid_hw // num_w_tiles
    pid_w = pid_hw % num_w_tiles

    ow_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)  # [BLOCK_W]
    oh_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    mask_w = ow_offs < OW
    mask_h = oh_offs < OH

    # accumulators: [BLOCK_H, OC, BLOCK_W]
    acc = tl.zeros([BLOCK_H, OC, BLOCK_W], dtype=tl.float32)

    # add conv bias
    oc_range = tl.arange(0, OC)
    cb = tl.load(cb_ptr + oc_range)  # [OC]
    acc = acc + cb[None, :, None]

    # Loop over kernel
    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_val = id_num // STRIDE
        id_valid = ((id_num % STRIDE) == 0) & (id_val >= 0) & (id_val < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh_offs + PAD - kh  # [BLOCK_H]
            ih_val = ih_num // STRIDE
            ih_valid = ((ih_num % STRIDE) == 0) & (ih_val >= 0) & (ih_val < IH) & mask_h  # [BLOCK_H]
            for kw in tl.static_range(0, KW):
                iw_num = ow_offs + PAD - kw  # [BLOCK_W]
                iw_val = iw_num // STRIDE
                iw_valid = ((iw_num % STRIDE) == 0) & (iw_val >= 0) & (iw_val < IW) & mask_w  # [BLOCK_W]
                # spatial_mask: [BLOCK_H, BLOCK_W]
                spatial_mask = id_valid & ih_valid[:, None] & iw_valid[None, :]

                for ic in tl.static_range(0, IC):
                    in_offset = (
                        pid_n * (IC * ID * IH * IW)
                        + ic * (ID * IH * IW)
                        + id_val * (IH * IW)
                        + ih_val[:, None] * IW
                        + iw_val[None, :]
                    )  # [BLOCK_H, BLOCK_W]
                    x_val = tl.load(x_ptr + in_offset, mask=spatial_mask, other=0.0)  # [BLOCK_H, BLOCK_W]

                    w_offset = (
                        ic * (OC * KD * KH * KW)
                        + oc_range * (KD * KH * KW)
                        + kd * (KH * KW)
                        + kh * KW
                        + kw
                    )
                    w_val = tl.load(w_ptr + w_offset)  # [OC]

                    # contribution: [BLOCK_H, OC, BLOCK_W] = w[None,:,None] * x[:,None,:]
                    acc = acc + w_val[None, :, None] * x_val[:, None, :]

    # Now acc is [BLOCK_H, OC, BLOCK_W]. Compute LSE across OC (axis=1).
    max_val = tl.max(acc, axis=1)  # [BLOCK_H, BLOCK_W]
    sum_exp = tl.sum(tl.exp(acc - max_val[:, None, :]), axis=1)  # [BLOCK_H, BLOCK_W]
    lse = max_val + tl.log(sum_exp)

    # HardSwish: x * sigmoid(x+3)/6
    hs = lse * tl.sigmoid(lse + 3.0) / 6.0

    b = tl.load(bias_ptr)
    out = hs - b
    out = tl.minimum(tl.maximum(out, -1.0), 1.0)

    out_offset = (
        pid_n * (OD * OH * OW)
        + od * (OH * OW)
        + oh_offs[:, None] * OW
        + ow_offs[None, :]
    )
    out_mask = mask_h[:, None] & mask_w[None, :]
    tl.store(out_ptr + out_offset, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        STRIDE = self.stride
        PAD = self.padding
        OC = self.out_channels

        OD = (ID - 1) * STRIDE - 2 * PAD + KD
        OH = (IH - 1) * STRIDE - 2 * PAD + KH
        OW = (IW - 1) * STRIDE - 2 * PAD + KW

        out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KD, KH, KW]
        cb = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        bias_flat = self.bias.contiguous().view(-1)[:1]

        grid = lambda META: (
            N, OD,
            ((OH + META['BLOCK_H'] - 1) // META['BLOCK_H']) *
            ((OW + META['BLOCK_W'] - 1) // META['BLOCK_W']),
        )

        fused_convtranspose3d_lse_hswish_sub_clamp_kernel[grid](
            x, weight, cb, bias_flat, out,
            N, OD, OH, OW,
            ID, IH, IW,
            IC, OC,
            KD, KH, KW,
            STRIDE, PAD,
        )
        return out