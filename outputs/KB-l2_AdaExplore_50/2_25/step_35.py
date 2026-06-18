import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SP': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SP': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SP': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'KH', 'KW'],
)
@triton.jit
def conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, H, W,
    OC: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oh, stride_ow,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < (OH * OW)
    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_offs = tl.arange(0, OC)

    acc = tl.zeros((OC, BLOCK_SP), dtype=tl.float32)

    x_base = pid_n * stride_xn

    # Loop ordering: kh, kw outer (static), ic inner (static)
    # Weights are tiny: OC*IC*KH*KW = 64*16*9 = 9216 floats
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh
            iw = ow + kw
            base_in = x_base + ih * stride_xh + iw * stride_xw
            for ic in tl.static_range(0, IC):
                x_off = base_in + ic * stride_xc
                x_val = tl.load(x_ptr + x_off, mask=sp_mask, other=0.0)
                # weight stride: OC contiguous in oc, layout is [OC, IC, KH, KW]
                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off)
                acc += w_val[:, None] * x_val[None, :]

    b_val = tl.load(b_ptr + oc_offs)
    acc += b_val[:, None]

    min_val = tl.min(acc, axis=0)

    y = tl.extra.cuda.libdevice.tanh(min_val)
    y = tl.extra.cuda.libdevice.tanh(y)

    out_off = pid_n * stride_on + oh * stride_oh + ow * stride_ow
    tl.store(out_ptr + out_off, y, mask=sp_mask)


def conv_min_tanh_tanh(x, weight, bias):
    N, IC, H, W = x.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (N, triton.cdiv(OH * OW, meta['BLOCK_SP']))

    conv_min_tanh_kernel[grid](
        x, weight, bias, out,
        N, IC, H, W, OC, KH, KW, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(2), out.stride(3),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.cuda().contiguous()
        b = self.conv.bias.cuda().contiguous()
        return conv_min_tanh_tanh(x, w, b)