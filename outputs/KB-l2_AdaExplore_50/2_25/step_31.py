import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SP': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 256}, num_warps=8, num_stages=3),
    ],
    key=['IC', 'OC', 'H', 'W', 'KH', 'KW'],
)
@triton.jit
def conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W, OC: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_on, stride_oh, stride_ow,
    BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < (OH * OW)
    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_offs = tl.arange(0, BLOCK_OC)
    ic_offs = tl.arange(0, BLOCK_IC)

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    x_base = x_ptr + pid_n * stride_xn + ic_offs[:, None] * stride_xc
    w_base = w_ptr + oc_offs[:, None] * stride_wo + ic_offs[None, :] * stride_wi

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh
            iw = ow + kw
            x_off = ih[None, :] * stride_xh + iw[None, :] * stride_xw
            x_tile = tl.load(x_base + x_off, mask=sp_mask[None, :], other=0.0)
            w_off = kh * stride_wh + kw * stride_ww
            w_tile = tl.load(w_base + w_off)
            acc += tl.dot(w_tile, x_tile, out_dtype=tl.float32)

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
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(2), out.stride(3),
        BLOCK_OC=OC, BLOCK_IC=IC,
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