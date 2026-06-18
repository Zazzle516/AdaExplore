import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 4, 'BLOCK_OW': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 8, 'BLOCK_OW': 16, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 8, 'BLOCK_OW': 32, 'BLOCK_IC': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 4, 'BLOCK_OW': 32, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 8, 'BLOCK_OW': 16, 'BLOCK_IC': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 4, 'BLOCK_OW': 16, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 4, 'BLOCK_OW': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 8, 'BLOCK_OW': 16, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'IC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv2d_nhwc_fused_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    constant_value, scaling_factor,
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_wo, stride_wh, stride_ww, stride_wi,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_OC: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    num_ow_tiles = (OW + BLOCK_OW - 1) // BLOCK_OW
    num_oh_tiles = (OH + BLOCK_OH - 1) // BLOCK_OH

    pid_oh = pid_sp // num_ow_tiles
    pid_ow = pid_sp % num_ow_tiles

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_oh = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)
    offs_ow = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)

    oc_mask = offs_oc < OC
    oh_mask = offs_oh < OH
    ow_mask = offs_ow < OW

    # Flatten spatial -> [BLOCK_OH*BLOCK_OW]
    # We'll compute acc as [BLOCK_OC, BLOCK_OH*BLOCK_OW] via dot.
    # Reduce over (kh, kw, ic_tile)
    BLOCK_SP: tl.constexpr = BLOCK_OH * BLOCK_OW

    # output spatial coords (flattened)
    oh_flat = tl.arange(0, BLOCK_OH)[:, None] + tl.zeros((1, BLOCK_OW), dtype=tl.int32)
    ow_flat = tl.arange(0, BLOCK_OW)[None, :] + tl.zeros((BLOCK_OH, 1), dtype=tl.int32)
    oh_idx = pid_oh * BLOCK_OH + oh_flat  # [BLOCK_OH, BLOCK_OW]
    ow_idx = pid_ow * BLOCK_OW + ow_flat
    sp_mask = (oh_idx < OH) & (ow_idx < OW)
    oh_idx_flat = tl.reshape(oh_idx, (BLOCK_SP,))
    ow_idx_flat = tl.reshape(ow_idx, (BLOCK_SP,))
    sp_mask_flat = tl.reshape(sp_mask, (BLOCK_SP,))

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Hoist constants out of inner loops
    oc_stride = offs_oc * stride_wo  # [BLOCK_OC]
    xn_base = pid_n * stride_xn

    for kh in range(0, KH):
        for kw in range(0, KW):
            ih = oh_idx_flat + kh
            iw = ow_idx_flat + kw
            sp_base = ih * stride_xh + iw * stride_xw  # [BLOCK_SP]
            w_khkw = kh * stride_wh + kw * stride_ww
            for ic_start in range(0, IC, BLOCK_IC):
                offs_ic = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = offs_ic < IC

                # weight: w[oc, kh, kw, ic] in NHWC weight layout
                w_ptrs = w_ptr + (oc_stride[:, None]
                                  + w_khkw
                                  + offs_ic[None, :] * stride_wi)
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_OC, BLOCK_IC]

                # input gather: x[n, ih, iw, ic] -- NHWC, contiguous on ic
                x_ptrs = x_ptr + (xn_base
                                  + sp_base[:, None]
                                  + offs_ic[None, :] * stride_xc)
                x_mask = sp_mask_flat[:, None] & ic_mask[None, :]
                x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_SP, BLOCK_IC]

                # acc += w_tile @ x_tile.T => [BLOCK_OC, BLOCK_SP]
                acc += tl.dot(w_tile, tl.trans(x_tile), allow_tf32=True)

    # conv bias
    b_vals = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc += b_vals[:, None]

    # min with constant
    acc = tl.minimum(acc, constant_value)

    # extra bias [OC]
    extra_bias = tl.load(bias_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc += extra_bias[:, None]

    # scale
    acc = acc * scaling_factor

    # store to NCHW output
    out_ptrs = out_ptr + (pid_n * stride_on
                          + offs_oc[:, None] * stride_oc
                          + oh_idx_flat[None, :] * stride_oh
                          + ow_idx_flat[None, :] * stride_ow)
    out_mask = oc_mask[:, None] & sp_mask_flat[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to NHWC layout: [OC, KH, KW, IC]
        with torch.no_grad():
            w_nhwc = self.conv.weight.detach().permute(0, 2, 3, 1).contiguous()
        self.register_buffer('weight_nhwc', w_nhwc)

    def forward(self, x):
        x = x.cuda()
        # NHWC input
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w_nhwc = self.weight_nhwc
        b = self.conv.bias.contiguous()
        bias_extra = self.bias.contiguous().view(-1)

        N, H, W, IC = x_nhwc.shape
        OC, KH, KW, _ = w_nhwc.shape
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            N,
            (OC + meta['BLOCK_OC'] - 1) // meta['BLOCK_OC'],
            ((OH + meta['BLOCK_OH'] - 1) // meta['BLOCK_OH']) *
            ((OW + meta['BLOCK_OW'] - 1) // meta['BLOCK_OW']),
        )

        conv2d_nhwc_fused_kernel[grid](
            x_nhwc, w_nhwc, b, bias_extra, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            float(self.constant_value), float(self.scaling_factor),
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            w_nhwc.stride(0), w_nhwc.stride(1), w_nhwc.stride(2), w_nhwc.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out