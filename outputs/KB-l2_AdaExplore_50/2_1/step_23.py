import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW', 'N'],
)
@triton.jit
def conv2d_relu_bias_kernel(
    x_ptr, w_ptr, conv_b_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # OC tile
    pid_nb = tl.program_id(1)  # combined (N * spatial) tile

    HW = OH * OW
    NHW = N * HW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_nb = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)

    nb = offs_nb // HW
    hw = offs_nb % HW
    oh = hw // OW
    ow = hw % OW

    K = IC * KH * KW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    KHW = KH * KW

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        k_mask = k < K

        ic = k // KHW
        rem = k % KHW
        kh = rem // KW
        kw = rem % KW

        w_ptrs = w_ptr + (offs_m[:, None] * stride_wo
                          + ic[None, :] * stride_wi
                          + kh[None, :] * stride_wkh
                          + kw[None, :] * stride_wkw)
        w_mask = (offs_m[:, None] < OC) & k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        ih = oh[None, :] + kh[:, None]
        iw = ow[None, :] + kw[:, None]

        x_ptrs = x_ptr + (nb[None, :] * stride_xn
                          + ic[:, None] * stride_xc
                          + ih * stride_xh
                          + iw * stride_xw)
        x_mask = k_mask[:, None] & (offs_nb[None, :] < NHW)
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(w_vals, x_vals)

    cb = tl.load(conv_b_ptr + offs_m, mask=offs_m < OC, other=0.0)
    acc += cb[:, None]

    acc = tl.maximum(acc, 0.0)

    eb = tl.load(bias_ptr + offs_m, mask=offs_m < OC, other=0.0)
    acc += eb[:, None]

    out_ptrs = out_ptr + (nb[None, :] * stride_on
                          + offs_m[:, None] * stride_oc
                          + oh[None, :] * stride_oh
                          + ow[None, :] * stride_ow)
    out_mask = (offs_m[:, None] < OC) & (offs_nb[None, :] < NHW)
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._eb_cache = None

    def _get_eb(self):
        if self._eb_cache is None or self._eb_cache.data_ptr() != self.bias.data_ptr():
            self._eb_cache = self.bias.view(-1).contiguous()
        return self._eb_cache

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()
        eb = self.bias.reshape(-1).contiguous()

        N, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        NHW = N * OH * OW
        grid = lambda meta: (
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(NHW, meta['BLOCK_N']),
        )

        conv2d_relu_bias_kernel[grid](
            x, w, cb, eb, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out