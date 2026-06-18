import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'OUT_HW', 'IC_KH_KW'],
)
@triton.jit
def conv_relu_bias_kernel(
    x_ptr, w_ptr, b_ptr, bias_add_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    OUT_HW,        # OH*OW
    IC_KH_KW,      # IC*KH*KW
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)              # batch
    pid_oc = tl.program_id(1)             # OC tile
    pid_hw = tl.program_id(2)             # output spatial tile

    offs_oc = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_hw = pid_hw * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    oh = offs_hw // OW
    ow = offs_hw % OW

    mask_oc = offs_oc < OC
    mask_hw = offs_hw < OUT_HW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for k_start in range(0, IC_KH_KW, BLOCK_K):
        k = k_start + offs_k                    # [BLOCK_K]
        mask_k = k < IC_KH_KW

        ic = k // (KH * KW)
        rem = k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # weight tile [BLOCK_M, BLOCK_K]
        w_offsets = (offs_oc[:, None] * stride_wo
                     + ic[None, :] * stride_wi
                     + kh[None, :] * stride_wh
                     + kw[None, :] * stride_ww)
        w_mask = mask_oc[:, None] & mask_k[None, :]
        w_tile = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        # input tile [BLOCK_K, BLOCK_N] via implicit im2col
        in_h = oh[None, :] + kh[:, None]   # [BLOCK_K, BLOCK_N]
        in_w = ow[None, :] + kw[:, None]   # [BLOCK_K, BLOCK_N]

        x_offsets = (pid_n * stride_xn
                     + ic[:, None] * stride_xc
                     + in_h * stride_xh
                     + in_w * stride_xw)
        x_mask = mask_k[:, None] & mask_hw[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        acc += tl.dot(w_tile, x_tile)

    # bias from conv
    b = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b[:, None]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # add learned bias (shape [OC, 1, 1])
    bias_add = tl.load(bias_add_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias_add[:, None]

    # store
    out_offsets = (pid_n * stride_on
                   + offs_oc[:, None] * stride_oc
                   + oh[None, :] * stride_oh
                   + ow[None, :] * stride_ow)
    out_mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()
        bias_add = self.bias.contiguous().view(-1)

        N, IC, H, W = x.shape
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        OUT_HW = OH * OW
        IC_KH_KW = IC * KH * KW

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(OUT_HW, meta['BLOCK_N']),
        )

        conv_relu_bias_kernel[grid](
            x, w, cb, bias_add, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            OUT_HW, IC_KH_KW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out