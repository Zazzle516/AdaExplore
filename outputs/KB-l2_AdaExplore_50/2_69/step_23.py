import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
    ],
    key=['N_OUT', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_hswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IH, IW,
    OC, OH, OW,
    IC: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    K_REAL: tl.constexpr, BLOCK_K: tl.constexpr,
    N_OUT,  # B * OH * OW
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k = tl.arange(0, BLOCK_K)

    mask_n = offs_n < N_OUT
    mask_oc = offs_oc < OC
    mask_k = offs_k < K_REAL

    # decode N axis -> (b, oh, ow)
    ow = offs_n % OW
    tmp = offs_n // OW
    oh = tmp % OH
    b = tmp // OH

    # decode K axis -> (ic, kh, kw)
    kw_idx = offs_k % KW
    tmp_k = offs_k // KW
    kh_idx = tmp_k % KH
    ic_idx = tmp_k // KH

    # Build X tile [BLOCK_N, BLOCK_K]
    x_base_n = b * stride_xb + oh * stride_xh + ow * stride_xw  # [BLOCK_N]
    x_k_off = ic_idx * stride_xc + kh_idx * stride_xh + kw_idx * stride_xw  # [BLOCK_K]
    x_off = x_base_n[:, None] + x_k_off[None, :]
    x_mask = mask_n[:, None] & mask_k[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

    # Build W tile [BLOCK_K, BLOCK_OC]
    w_off = (ic_idx * stride_wi + kh_idx * stride_wh + kw_idx * stride_ww)[:, None] + offs_oc[None, :] * stride_wo
    w_mask = mask_k[:, None] & mask_oc[None, :]
    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

    acc = tl.dot(x_tile, w_tile, out_dtype=tl.float32)

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    # hardswish then relu: y = relu(x * relu6(x+3)/6)
    # since relu(hardswish(x)) = hardswish(x) for x>=0, and 0 otherwise (hardswish(x)<=0 when x<=0)
    # actually hardswish(x) = 0 when x<=-3, negative for -3<x<0, positive for x>0
    # relu(hardswish(x)) = x*(x+3)/6 clamped: for x>=3 -> x; for 0<=x<3 -> x*(x+3)/6; else 0
    x_in = acc
    relu6 = tl.minimum(tl.maximum(x_in + 3.0, 0.0), 6.0)
    hs = x_in * relu6 * (1.0 / 6.0)
    out_val = tl.maximum(hs, 0.0)

    out_off = (b * stride_ob + oh * stride_oh + ow * stride_ow)[:, None] + offs_oc[None, :] * stride_oc
    mask = mask_n[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_off, out_val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.cuda().contiguous()
        b = self.conv.bias.cuda().contiguous()

        B, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)

        N_OUT = B * OH * OW

        grid = lambda meta: (
            triton.cdiv(N_OUT, meta['BLOCK_N']),
            triton.cdiv(OC, meta['BLOCK_OC']),
        )

        K_REAL = IC * KH * KW
        # pad K to next pow2 >= 16
        BLOCK_K = 1
        while BLOCK_K < K_REAL:
            BLOCK_K *= 2
        if BLOCK_K < 16:
            BLOCK_K = 16

        conv_hswish_relu_kernel[grid](
            x, w, b, out,
            B, IH, IW,
            OC, OH, OW,
            IC, KH, KW,
            K_REAL, BLOCK_K,
            N_OUT,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out