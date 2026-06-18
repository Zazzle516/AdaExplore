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
    key=['IC', 'OC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, H, W,
    OC: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OH, OW,
    stride_xn, stride_xh, stride_xw,  # NHWC strides for x (channel stride = 1)
    stride_on, stride_oh, stride_ow,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
    K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < (OH * OW)
    oh = sp_offs // OW
    ow = sp_offs % OW

    oc_offs = tl.arange(0, BLOCK_OC)
    k_offs = tl.arange(0, K)  # K = KH*KW*IC

    # Decompose k into (kh, kw, ic) using IC contiguous
    ic_idx = k_offs % IC
    khw = k_offs // IC
    kw_idx = khw % KW
    kh_idx = khw // KW

    # weight is pre-packed as [OC, K] contiguous
    w_off = oc_offs[:, None] * K + k_offs[None, :]
    w_tile = tl.load(w_ptr + w_off)  # [OC, K]

    # x in NHWC: offset = n*stride_xn + (oh+kh)*stride_xh + (ow+kw)*stride_xw + ic
    # We gather x as [K, BLOCK_SP]
    ih = oh[None, :] + kh_idx[:, None]  # [K, SP]
    iw = ow[None, :] + kw_idx[:, None]  # [K, SP]
    x_off = (pid_n * stride_xn
             + ih * stride_xh
             + iw * stride_xw
             + ic_idx[:, None])
    x_mask = sp_mask[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [K, SP]

    acc = tl.dot(w_tile, x_tile)  # [OC, SP]

    b_val = tl.load(b_ptr + oc_offs)
    acc += b_val[:, None]

    min_val = tl.min(acc, axis=0)

    y = tl.extra.cuda.libdevice.tanh(min_val)
    y = tl.extra.cuda.libdevice.tanh(y)

    out_off = pid_n * stride_on + oh * stride_oh + ow * stride_ow
    tl.store(out_ptr + out_off, y, mask=sp_mask)


def conv_min_tanh_tanh(x_nhwc, weight_packed, bias, N, IC, H, W, OC, KH, KW):
    OH = H - KH + 1
    OW = W - KW + 1

    out = torch.empty((N, 1, OH, OW), device=x_nhwc.device, dtype=x_nhwc.dtype)

    K = KH * KW * IC

    grid = lambda meta: (N, triton.cdiv(OH * OW, meta['BLOCK_SP']))

    conv_min_tanh_kernel[grid](
        x_nhwc, weight_packed, bias, out,
        N, IC, H, W, OC, KH, KW, OH, OW,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2),
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

    def forward(self, x):
        x = x.cuda().contiguous()
        N, IC, H, W = x.shape
        # Convert input to NHWC layout (IC contiguous)
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w = self.conv.weight.cuda().contiguous()  # [OC, IC, KH, KW]
        OC, _, KH, KW = w.shape
        # Pack weight: [OC, KH, KW, IC] -> flat [OC, KH*KW*IC]
        w_packed = w.permute(0, 2, 3, 1).contiguous().view(OC, KH * KW * IC)

        b = self.conv.bias.cuda().contiguous()

        return conv_min_tanh_tanh(x_nhwc, w_packed, b, N, IC, H, W, OC, KH, KW)