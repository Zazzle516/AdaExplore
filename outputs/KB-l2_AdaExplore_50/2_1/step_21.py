import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_relu_bias_nhwc_kernel(
    x_ptr, w_ptr, conv_b_ptr, bias_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    # x is NHWC: strides
    stride_xn, stride_xh, stride_xw, stride_xc,
    # w is OC, KH, KW, IC contiguous
    stride_wo, stride_wkh, stride_wkw, stride_wi,
    # out is NHWC
    stride_on, stride_oh, stride_ow, stride_oc,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # OC tile
    pid_n = tl.program_id(1)  # spatial tile (OH*OW)
    pid_b = tl.program_id(2)  # batch

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial indices

    oh = offs_n // OW
    ow = offs_n % OW

    spatial_mask = offs_n < (OH * OW)
    oc_mask = offs_m < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_ic = tl.arange(0, IC)  # full IC reduction

    # Loop over kernel positions; inner reduction is over IC (BLOCK_K=IC)
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # Weight tile: [BLOCK_M, IC]
            w_ptrs = w_ptr + (offs_m[:, None] * stride_wo
                              + kh * stride_wkh
                              + kw * stride_wkw
                              + offs_ic[None, :] * stride_wi)
            w_vals = tl.load(w_ptrs, mask=oc_mask[:, None], other=0.0)

            # Input tile: [IC, BLOCK_N]
            ih = oh + kh  # [BLOCK_N]
            iw = ow + kw
            x_ptrs = x_ptr + (pid_b * stride_xn
                              + ih[None, :] * stride_xh
                              + iw[None, :] * stride_xw
                              + offs_ic[:, None] * stride_xc)
            x_vals = tl.load(x_ptrs, mask=spatial_mask[None, :], other=0.0)

            acc += tl.dot(w_vals, x_vals)

    # add conv bias
    cb = tl.load(conv_b_ptr + offs_m, mask=oc_mask, other=0.0)
    acc += cb[:, None]
    # ReLU
    acc = tl.maximum(acc, 0.0)
    # add extra bias
    eb = tl.load(bias_ptr + offs_m, mask=oc_mask, other=0.0)
    acc += eb[:, None]

    # Store as NHWC
    out_ptrs = out_ptr + (pid_b * stride_on
                          + oh[None, :] * stride_oh
                          + ow[None, :] * stride_ow
                          + offs_m[:, None] * stride_oc)
    out_mask = oc_mask[:, None] & spatial_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Cache reformatted weight (OC, KH, KW, IC) contiguous
        self._cached_w = None

    def _get_weight_nhwc(self):
        w = self.conv.weight  # (OC, IC, KH, KW)
        # Permute to (OC, KH, KW, IC) and make contiguous
        w_perm = w.permute(0, 2, 3, 1).contiguous()
        return w_perm

    def forward(self, x):
        # x: (N, IC, H, W) -> NHWC (N, H, W, IC)
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w_nhwc = self._get_weight_nhwc()
        cb = self.conv.bias.contiguous()
        eb = self.bias.view(-1).contiguous()

        N, IH, IW, IC = x_nhwc.shape
        OC, KH, KW, _ = w_nhwc.shape
        OH = IH - KH + 1
        OW = IW - KW + 1

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(OH * OW, meta['BLOCK_N']),
            N,
        )

        conv2d_relu_bias_nhwc_kernel[grid](
            x_nhwc, w_nhwc, cb, eb, out_nhwc,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            w_nhwc.stride(0), w_nhwc.stride(1), w_nhwc.stride(2), w_nhwc.stride(3),
            out_nhwc.stride(0), out_nhwc.stride(1), out_nhwc.stride(2), out_nhwc.stride(3),
        )

        # Convert back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out