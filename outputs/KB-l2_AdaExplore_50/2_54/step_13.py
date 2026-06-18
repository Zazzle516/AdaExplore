import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_HW': 256}, num_warps=4, num_stages=3),
    ],
    key=['C_in', 'C_out', 'H_out', 'W_out'],
)
@triton.jit
def conv2d_fused_kernel(
    x_ptr, w_ptr, b_ptr, m_ptr, out_ptr,
    N, C_in, H, W,
    C_out, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oh = hw_offs // W_out
    ow = hw_offs % W_out

    oc_mask = oc_offs < C_out
    hw_mask = hw_offs < (H_out * W_out)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    x_base = pid_n * stride_xn + oh * stride_xh + ow * stride_xw  # [BLOCK_HW]
    w_base = oc_offs * stride_wo  # [BLOCK_OC]

    for ic in range(0, C_in):
        x_ic = x_base + ic * stride_xc
        w_ic = w_base + ic * stride_wi
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                x_offsets = x_ic + kh * stride_xh + kw * stride_xw
                x_vals = tl.load(x_ptr + x_offsets, mask=hw_mask, other=0.0)
                w_offsets = w_ic + kh * stride_wkh + kw * stride_wkw
                w_vals = tl.load(w_ptr + w_offsets, mask=oc_mask, other=0.0)
                acc += w_vals[:, None] * x_vals[None, :]

    # bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_vals[:, None]

    # multiplier (per out_channel)
    m_vals = tl.load(m_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc * m_vals[:, None]

    # LeakyReLU (default negative_slope=0.01)
    acc = tl.where(acc >= 0, acc, acc * 0.01)

    # GELU (erf-based exact): 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # store
    out_offsets = (pid_n * stride_on +
                   oc_offs[:, None] * stride_oc +
                   oh[None, :] * stride_oh +
                   ow[None, :] * stride_ow)
    mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()
        m = self.multiplier.contiguous().cuda().view(-1)

        N, C_in, H, W = x.shape
        C_out, _, KH, KW = w.shape
        H_out = H - KH + 1
        W_out = W - KW + 1

        out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

        grid = lambda META: (
            N,
            triton.cdiv(C_out, META['BLOCK_OC']),
            triton.cdiv(H_out * W_out, META['BLOCK_HW']),
        )

        conv2d_fused_kernel[grid](
            x, w, b, m, out,
            N, C_in, H, W,
            C_out, H_out, W_out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            KH=KH, KW=KW,
        )
        return out