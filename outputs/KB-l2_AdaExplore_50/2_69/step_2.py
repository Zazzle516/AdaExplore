import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
    ],
    key=['N_OUT', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_hswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC: tl.constexpr, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
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

    mask_n = offs_n < N_OUT
    mask_oc = offs_oc < OC

    # decode N axis -> (b, oh, ow)
    ow = offs_n % OW
    tmp = offs_n // OW
    oh = tmp % OH
    b = tmp // OH

    acc = tl.zeros((BLOCK_N, BLOCK_OC), dtype=tl.float32)

    # Hoist invariant base offsets
    x_bh_base = b * stride_xb  # [BLOCK_N]
    w_oc_base = offs_oc * stride_wo  # [BLOCK_OC]

    for ic in tl.static_range(0, IC):
        x_ic = x_bh_base + ic * stride_xc
        w_ic = w_oc_base + ic * stride_wi
        for kh in tl.static_range(0, KH):
            ih = oh + kh
            x_ich = x_ic + ih * stride_xh
            w_ich = w_ic + kh * stride_wh
            for kw in tl.static_range(0, KW):
                iw = ow + kw
                x_off = x_ich + iw * stride_xw
                x_vals = tl.load(x_ptr + x_off, mask=mask_n, other=0.0)  # [BLOCK_N]

                w_off = w_ich + kw * stride_ww
                w_vals = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                acc += x_vals[:, None] * w_vals[None, :]

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

        conv_hswish_relu_kernel[grid](
            x, w, b, out,
            B, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            N_OUT,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out