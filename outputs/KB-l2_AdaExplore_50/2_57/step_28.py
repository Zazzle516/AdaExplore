import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 32, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 16, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 32, 'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64, 'BLOCK_OC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 32, 'BLOCK_OC': 64}, num_warps=4, num_stages=4),
    ],
    key=['OH', 'OW', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_relu_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # Grid: (cdiv(OH, BLOCK_OH) * cdiv(OW, BLOCK_OW), cdiv(OC, BLOCK_OC), B)
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_b = tl.program_id(2)

    num_ow_blocks = tl.cdiv(OW, BLOCK_OW)
    pid_oh = pid_sp // num_ow_blocks
    pid_ow = pid_sp % num_ow_blocks

    offs_oh = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)
    offs_ow = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    mask_oh = offs_oh < OH
    mask_ow = offs_ow < OW
    mask_oc = offs_oc < OC

    # accumulator [BLOCK_OH, BLOCK_OW, BLOCK_OC] flattened as [BLOCK_OH*BLOCK_OW, BLOCK_OC]
    acc = tl.zeros((BLOCK_OH * BLOCK_OW, BLOCK_OC), dtype=tl.float32)

    # spatial mask flattened
    sp_mask = (mask_oh[:, None] & mask_ow[None, :])
    sp_mask_flat = tl.reshape(sp_mask, (BLOCK_OH * BLOCK_OW,))

    b_off = pid_b * stride_xb

    for ic in range(0, IC):
        ic_x_off = ic * stride_xc
        ic_w_off = ic * stride_wi
        for kh in tl.static_range(0, KH):
            ih = offs_oh + kh  # [BLOCK_OH]
            for kw in tl.static_range(0, KW):
                iw = offs_ow + kw  # [BLOCK_OW]
                # x: [BLOCK_OH, BLOCK_OW]
                x_offs = b_off + ic_x_off + ih[:, None] * stride_xh + iw[None, :] * stride_xw
                x_vals = tl.load(x_ptr + x_offs, mask=sp_mask, other=0.0)
                x_flat = tl.reshape(x_vals, (BLOCK_OH * BLOCK_OW,))

                # w: [BLOCK_OC]
                w_offs = offs_oc * stride_wo + ic_w_off + kh * stride_wh + kw * stride_ww
                w_vals = tl.load(w_ptr + w_offs, mask=mask_oc, other=0.0)

                acc += x_flat[:, None] * w_vals[None, :]

    # bias
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    # relu
    acc = tl.maximum(acc, 0.0)

    # hardswish: x * clamp((x+3)/6, 0, 1)
    hs = (acc + 3.0) * (1.0 / 6.0)
    hs = tl.minimum(tl.maximum(hs, 0.0), 1.0)
    acc = acc * hs

    # store: reshape back to [BLOCK_OH, BLOCK_OW, BLOCK_OC]
    out_3d = tl.reshape(acc, (BLOCK_OH, BLOCK_OW, BLOCK_OC))
    out_offs = (pid_b * stride_ob +
                offs_oc[None, None, :] * stride_oc +
                offs_oh[:, None, None] * stride_oh +
                offs_ow[None, :, None] * stride_ow)
    out_mask = mask_oh[:, None, None] & mask_ow[None, :, None] & mask_oc[None, None, :]
    tl.store(out_ptr + out_offs, out_3d, mask=out_mask)


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

        B, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(OH, meta['BLOCK_OH']) * triton.cdiv(OW, meta['BLOCK_OW']),
            triton.cdiv(OC, meta['BLOCK_OC']),
            B,
        )

        conv_relu_hardswish_kernel[grid](
            x, w, b, out,
            B, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out