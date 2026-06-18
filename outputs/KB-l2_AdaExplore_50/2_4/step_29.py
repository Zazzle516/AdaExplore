import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mish(x):
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    t = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    return x * t


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['N', 'IC', 'OC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv2d_mish_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    # strides for x (N, IC, IH, IW) NCHW
    sx_n, sx_c, sx_h, sx_w,
    # strides for w (OC, IC, KH, KW)
    sw_o, sw_c, sw_h, sw_w,
    # strides for out (N, OC, OH, OW)
    so_n, so_c, so_h, so_w,
    BLOCK_M: tl.constexpr,  # output spatial tile
    BLOCK_N: tl.constexpr,  # output channel tile
    BLOCK_K: tl.constexpr,  # IC reduction tile
):
    pid_n = tl.program_id(0)  # batch
    pid_m = tl.program_id(1)  # spatial tile (over OH*OW)
    pid_oc = tl.program_id(2)  # OC tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial positions
    offs_oc = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # output channels

    oh = offs_m // OW
    ow = offs_m % OW
    m_mask = offs_m < (OH * OW)
    n_mask = offs_oc < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kernel positions and IC blocks
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # no padding
            iw = ow + kw
            # Loop over IC in chunks
            for ic_start in range(0, IC, BLOCK_K):
                offs_k = ic_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < IC

                # Load x[n, offs_k, ih, iw]: shape (BLOCK_M, BLOCK_K)
                x_offs = (pid_n * sx_n
                          + offs_k[None, :] * sx_c
                          + ih[:, None] * sx_h
                          + iw[:, None] * sx_w)
                x_mask = m_mask[:, None] & k_mask[None, :]
                x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

                # Load w[offs_oc, offs_k, kh, kw]: shape (BLOCK_K, BLOCK_N)
                w_offs = (offs_oc[None, :] * sw_o
                          + offs_k[:, None] * sw_c
                          + kh * sw_h
                          + kw * sw_w)
                w_mask = k_mask[:, None] & n_mask[None, :]
                w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Bias
    b = tl.load(b_ptr + offs_oc, mask=n_mask, other=0.0)
    acc += b[None, :]

    # Double mish
    acc = _mish(acc)
    acc = _mish(acc)

    # Store
    out_offs = (pid_n * so_n
                + offs_oc[None, :] * so_c
                + oh[:, None] * so_h
                + ow[:, None] * so_w)
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv2d_mish_mish(x, w, b):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        N,
        triton.cdiv(OH * OW, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    conv2d_mish_mish_kernel[grid](
        x, w, b, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        return conv2d_mish_mish(x, w, b)