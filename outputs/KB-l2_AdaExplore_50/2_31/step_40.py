import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['N', 'OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_min_bias_scale_kernel(
    x_ptr, w_ptr, b_conv_ptr, b_extra_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    constant_value, scaling_factor,
    # x is NHWC: stride (IC*IH*IW, IW*IC, IC, 1)
    stride_xn, stride_xh, stride_xw, stride_xc,
    # w is (OC, KH, KW, IC): stride (KH*KW*IC, KW*IC, IC, 1)
    stride_woc, stride_wkh, stride_wkw, stride_wic,
    # out is NHWC: (N, OH, OW, OC)
    stride_on, stride_oh, stride_ow, stride_oc,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # over (N * OH * OW) tiles
    pid_n = tl.program_id(1)  # over OC tiles

    M = N * OH * OW
    K = KH * KW * IC

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # decompose offs_m into (n, oh, ow)
    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    m_mask = offs_m < M
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k  # [BLOCK_K]
        k_mask = k < K

        # decompose k into (kh, kw, ic)
        kh = k // (KW * IC)
        kw_ic = k % (KW * IC)
        kw = kw_ic // IC
        ic = kw_ic % IC

        # input H,W positions for each m and each k
        ih = oh_idx[:, None] + kh[None, :]  # [BLOCK_M, BLOCK_K]
        iw = ow_idx[:, None] + kw[None, :]

        x_offsets = (n_idx[:, None] * stride_xn
                     + ih * stride_xh
                     + iw * stride_xw
                     + ic[None, :] * stride_xc)

        x_load_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=x_load_mask, other=0.0)

        # weight tile: (BLOCK_K, BLOCK_N)
        w_offsets = (offs_n[None, :] * stride_woc
                     + kh[:, None] * stride_wkh
                     + kw[:, None] * stride_wkw
                     + ic[:, None] * stride_wic)
        w_load_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_ptr + w_offsets, mask=w_load_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile)

    # add conv bias
    b_conv = tl.load(b_conv_ptr + offs_n, mask=n_mask, other=0.0)
    acc += b_conv[None, :]

    # min with constant
    acc = tl.minimum(acc, constant_value)

    # add extra bias (per-channel)
    b_extra = tl.load(b_extra_ptr + offs_n, mask=n_mask, other=0.0)
    acc += b_extra[None, :]

    # scale
    acc *= scaling_factor

    # store NHWC
    out_offsets = (n_idx[:, None] * stride_on
                   + oh_idx[:, None] * stride_oh
                   + ow_idx[:, None] * stride_ow
                   + offs_n[None, :] * stride_oc)
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.constant_value = float(constant_value)
        self.scaling_factor = float(scaling_factor)

        # Same init as nn.Conv2d
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # weight: (OC, IC, KH, KW) -> permute to (OC, KH, KW, IC)
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.conv_bias = nn.Parameter(conv.bias.detach().clone())
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        # to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        # weight (OC, IC, KH, KW) -> (OC, KH, KW, IC)
        w = self.weight.permute(0, 2, 3, 1).contiguous()

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        b_extra = self.bias.view(-1).contiguous()

        M = N * OH * OW
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )

        conv2d_min_bias_scale_kernel[grid](
            x_nhwc, w, self.conv_bias, b_extra, out_nhwc,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            self.constant_value, self.scaling_factor,
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            out_nhwc.stride(0), out_nhwc.stride(1), out_nhwc.stride(2), out_nhwc.stride(3),
        )

        # back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out