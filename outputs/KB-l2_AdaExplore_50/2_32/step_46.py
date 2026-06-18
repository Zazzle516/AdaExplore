import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 2, 'BLOCK_OW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'IC', 'KH', 'KW', 'OH', 'OW'],
)
@triton.jit
def conv_scale_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W, OC, KH, KW, OH, OW,
    scale,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    BLOCK_OC: tl.constexpr,
    BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    offs_oh = pid_h * BLOCK_OH + tl.arange(0, BLOCK_OH)  # [BLOCK_OH]
    offs_ow = pid_w * BLOCK_OW + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]
    mask_oh = offs_oh < OH
    mask_ow = offs_ow < OW

    offs_oc = tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    mask_oc = offs_oc < OC

    # Accumulator [BLOCK_OC, BLOCK_OH, BLOCK_OW]
    acc = tl.zeros([BLOCK_OC, BLOCK_OH, BLOCK_OW], dtype=tl.float32)

    x_batch_ptr = x_ptr + pid_n * stride_xn

    for ic in range(0, IC):
        for kh in range(0, KH):
            for kw in range(0, KW):
                # Load weight slice w[:, ic, kh, kw] -> [BLOCK_OC]
                w_ptrs = w_ptr + offs_oc * stride_wo + ic * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vals = tl.load(w_ptrs, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                # Load x slice [BLOCK_OH, BLOCK_OW] from x[n, ic, offs_oh+kh, offs_ow+kw]
                ih = offs_oh + kh  # [BLOCK_OH]
                iw = offs_ow + kw  # [BLOCK_OW]
                x_ptrs = (x_batch_ptr
                          + ic * stride_xc
                          + ih[:, None] * stride_xh
                          + iw[None, :] * stride_xw)
                x_mask = mask_oh[:, None] & mask_ow[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_OH, BLOCK_OW]

                # Outer product: w[BLOCK_OC] * x[BLOCK_OH, BLOCK_OW] -> [BLOCK_OC, BLOCK_OH, BLOCK_OW]
                acc += w_vals[:, None, None] * x_vals[None, :, :]

    # Add bias
    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b_vals[:, None, None]
    acc = acc * scale

    # Mask invalid oc rows with +inf
    acc = tl.where(mask_oc[:, None, None], acc, float('inf'))

    # Reduce min along OC axis -> [BLOCK_OH, BLOCK_OW]
    min_val = tl.min(acc, axis=0)

    # Store
    out_offsets = (pid_n * OH * OW
                   + offs_oh[:, None] * OW
                   + offs_ow[None, :])
    out_mask = mask_oh[:, None] & mask_ow[None, :]
    tl.store(out_ptr + out_offsets, min_val, mask=out_mask)


def conv_scale_min(x, weight, bias, scale):
    N, IC, H, W = x.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty((N, 1, OH, OW), device=x.device, dtype=torch.float32)

    # BLOCK_OC must be a power of 2 >= OC
    BLOCK_OC = 1
    while BLOCK_OC < OC:
        BLOCK_OC *= 2

    grid = lambda meta: (N, triton.cdiv(OH, meta['BLOCK_OH']), triton.cdiv(OW, meta['BLOCK_OW']))

    conv_scale_min_kernel[grid](
        x, weight, bias, out,
        N, IC, H, W, OC, KH, KW, OH, OW,
        scale,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        BLOCK_OC=BLOCK_OC,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()
        return conv_scale_min(x, w, b, float(self.scale_factor))