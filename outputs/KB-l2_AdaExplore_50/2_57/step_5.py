import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
    ],
    key=['N_OUT', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_relu_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, IC: tl.constexpr, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    N_OUT,
    stride_xb, stride_xh, stride_xw,  # NHWC strides for x (IC contiguous = 1)
    stride_wo,  # w is reshaped to [OC, KH*KW*IC] contiguous
    stride_yb, stride_yh, stride_yw,  # NHWC strides for y
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    mask_n = offs_n < N_OUT
    mask_oc = offs_oc < OC

    ow = offs_n % OW
    tmp = offs_n // OW
    oh = tmp % OH
    b = tmp // OH

    K = KH * KW * IC  # reduction dim
    offs_ic = tl.arange(0, IC)  # IC must be power of 2 or small constexpr

    acc = tl.zeros((BLOCK_N, BLOCK_OC), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # [BLOCK_N]
            iw = ow + kw  # [BLOCK_N]
            # x: NHWC, IC contiguous. addr = b*stride_xb + ih*stride_xh + iw*stride_xw + ic
            x_base = b * stride_xb + ih * stride_xh + iw * stride_xw  # [BLOCK_N]
            x_offs = x_base[:, None] + offs_ic[None, :]  # [BLOCK_N, IC]
            x_tile = tl.load(x_ptr + x_offs, mask=mask_n[:, None], other=0.0)  # [BLOCK_N, IC]

            # w: [OC, KH, KW, IC] reshaped to [OC, KH*KW*IC]; k = (kh*KW + kw)*IC + ic
            k_base = (kh * KW + kw) * IC
            w_offs = offs_oc[:, None] * stride_wo + (k_base + offs_ic[None, :])  # [BLOCK_OC, IC]
            w_tile = tl.load(w_ptr + w_offs, mask=mask_oc[:, None], other=0.0)  # [BLOCK_OC, IC]

            # acc += x_tile @ w_tile^T  => [BLOCK_N, BLOCK_OC]
            acc += tl.dot(x_tile, tl.trans(w_tile))

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)
    # HardSwish on relu output: x * clamp((x+3)/6, 0, 1)
    hs = (acc + 3.0) / 6.0
    hs = tl.minimum(tl.maximum(hs, 0.0), 1.0)
    out = acc * hs

    # store to y in NHWC layout
    y_offs = (b[:, None] * stride_yb + oh[:, None] * stride_yh +
              ow[:, None] * stride_yw + offs_oc[None, :])
    mask = mask_n[:, None] & mask_oc[None, :]
    tl.store(y_ptr + y_offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-transpose weight to [OC, KH, KW, IC] layout, contiguous
        with torch.no_grad():
            w = self.conv.weight.detach()  # [OC, IC, KH, KW]
            w_nhwc = w.permute(0, 2, 3, 1).contiguous()  # [OC, KH, KW, IC]
        self.register_buffer('w_packed', w_nhwc.cuda())
        self.register_buffer('b_packed', self.conv.bias.detach().contiguous().cuda())

    def forward(self, x):
        x = x.cuda()
        B, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        # Convert x to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [B, IH, IW, IC]
        # Output in NHWC
        y_nhwc = torch.empty((B, OH, OW, OC), device=x.device, dtype=x.dtype)

        N_OUT = B * OH * OW

        # x strides (in elements) for NHWC
        sxb = IH * IW * IC
        sxh = IW * IC
        sxw = IC
        # y strides for NHWC
        syb = OH * OW * OC
        syh = OW * OC
        syw = OC
        # weight stride along OC axis (packed [OC, KH*KW*IC])
        swo = KH * KW * IC

        grid = lambda meta: (
            triton.cdiv(N_OUT, meta['BLOCK_N']),
            triton.cdiv(OC, meta['BLOCK_OC']),
        )

        conv_relu_hardswish_kernel[grid](
            x_nhwc, self.w_packed, self.b_packed, y_nhwc,
            B, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            N_OUT,
            sxb, sxh, sxw,
            swo,
            syb, syh, syw,
        )

        # Convert back to NCHW
        y = y_nhwc.permute(0, 3, 1, 2).contiguous()
        return y