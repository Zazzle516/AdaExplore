import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SP': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 512}, num_warps=8, num_stages=3),
    ],
    key=['IC', 'OC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, H, W,
    OC: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OH, OW,
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_on, stride_oh, stride_ow,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr, K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < (OH * OW)
    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_offs = tl.arange(0, BLOCK_OC)

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    ic_range = tl.arange(0, IC)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh
            iw = ow + kw
            x_off = (pid_n * stride_xn
                     + ih[None, :] * stride_xh
                     + iw[None, :] * stride_xw
                     + ic_range[:, None] * stride_xc)
            x_tile = tl.load(x_ptr + x_off, mask=sp_mask[None, :], other=0.0)
            kk = (kh * KW + kw) * IC
            w_off = oc_offs[:, None] * K + (kk + ic_range[None, :])
            w_tile = tl.load(w_ptr + w_off)
            acc += tl.dot(w_tile, x_tile, out_dtype=tl.float32)

    b_val = tl.load(b_ptr + oc_offs)
    acc += b_val[:, None]

    min_val = tl.min(acc, axis=0)

    y = tl.extra.cuda.libdevice.tanh(min_val)
    y = tl.extra.cuda.libdevice.tanh(y)

    out_off = pid_n * stride_on + oh * stride_oh + ow * stride_ow
    tl.store(out_ptr + out_off, y, mask=sp_mask)


def conv_min_tanh_tanh(x_nhwc, weight_packed, bias, OC, IC, KH, KW, N, H, W):
    OH = H - KH + 1
    OW = W - KW + 1

    out = torch.empty((N, 1, OH, OW), device=x_nhwc.device, dtype=x_nhwc.dtype)

    K = IC * KH * KW

    grid = lambda meta: (N, triton.cdiv(OH * OW, meta['BLOCK_SP']))

    conv_min_tanh_kernel[grid](
        x_nhwc, weight_packed, bias, out,
        N, IC, H, W, OC, KH, KW, OH, OW,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        out.stride(0), out.stride(2), out.stride(3),
        BLOCK_OC=OC, K=K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._packed_weight = None
        self._packed_bias = None

    def _pack_weight(self, device):
        w = self.conv.weight.detach().to(device).contiguous()
        OC, IC, KH, KW = w.shape
        wp = w.permute(0, 2, 3, 1).contiguous().view(OC, KH * KW * IC).contiguous()
        return wp

    def forward(self, x):
        x = x.cuda()
        if (self._packed_weight is None
                or self._packed_weight.device != x.device):
            self._packed_weight = self._pack_weight(x.device)
            self._packed_bias = self.conv.bias.detach().to(x.device).contiguous()
        OC = self.out_channels
        IC = self.in_channels
        KH = self.kernel_size
        KW = self.kernel_size
        N, _, H, W = x.shape
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        return conv_min_tanh_tanh(x_nhwc, self._packed_weight, self._packed_bias,
                                   OC, IC, KH, KW, N, H, W)