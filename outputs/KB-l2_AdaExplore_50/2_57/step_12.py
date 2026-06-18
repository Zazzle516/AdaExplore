import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 128}, num_warps=8, num_stages=3),
    ],
    key=['N_OUT', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_relu_hardswish_nhwc_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, IC: tl.constexpr, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    N_OUT,  # B*OH*OW
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_ic = tl.arange(0, IC)  # IC is constexpr (8)

    mask_n = offs_n < N_OUT
    mask_oc = offs_oc < OC

    # decompose n -> (b, oh, ow)
    ow = offs_n % OW
    tmp = offs_n // OW
    oh = tmp % OH
    b = tmp // OH

    # NHWC strides for x: (IH*IW*IC, IW*IC, IC, 1)
    # NHWC weight layout: (OC, KH, KW, IC), strides: (KH*KW*IC, KW*IC, IC, 1)
    x_base = b * (IH * IW * IC)  # [BLOCK_N]

    acc = tl.zeros((BLOCK_N, BLOCK_OC), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh
            iw = ow + kw
            # x offsets: x_base + ih*IW*IC + iw*IC + ic
            x_offs = (x_base + ih * (IW * IC) + iw * IC)[:, None] + offs_ic[None, :]
            x_tile = tl.load(x_ptr + x_offs, mask=mask_n[:, None], other=0.0)  # [BLOCK_N, IC]

            # w offsets: oc*(KH*KW*IC) + kh*(KW*IC) + kw*IC + ic
            w_offs = (offs_oc[:, None] * (KH * KW * IC) +
                      (kh * KW * IC + kw * IC) + offs_ic[None, :])
            w_tile = tl.load(w_ptr + w_offs, mask=mask_oc[:, None], other=0.0)  # [BLOCK_OC, IC]

            acc += tl.dot(x_tile, tl.trans(w_tile))

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)
    # HardSwish: x * clamp((x+3)/6, 0, 1)
    hs = (acc + 3.0) * (1.0 / 6.0)
    hs = tl.minimum(tl.maximum(hs, 0.0), 1.0)
    out = acc * hs

    # store NHWC: y_offs = b*(OH*OW*OC) + oh*OW*OC + ow*OC + oc
    y_base = b * (OH * OW * OC) + oh * (OW * OC) + ow * OC
    y_offs = y_base[:, None] + offs_oc[None, :]
    mask = mask_n[:, None] & mask_oc[None, :]
    tl.store(y_ptr + y_offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Pre-permute weight to NHWC-like layout (OC, KH, KW, IC)
        with torch.no_grad():
            w = self.conv.weight.detach().contiguous()  # (OC, IC, KH, KW)
            w_nhwc = w.permute(0, 2, 3, 1).contiguous()  # (OC, KH, KW, IC)
        self.register_buffer('w_nhwc', w_nhwc)
        self.register_buffer('bias_buf', self.conv.bias.detach().contiguous())

    def forward(self, x):
        x = x.contiguous().cuda()
        B, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        # convert x to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (B, IH, IW, IC)

        w = self.w_nhwc.to(x.device)
        b = self.bias_buf.to(x.device)

        y_nhwc = torch.empty((B, OH, OW, OC), device=x.device, dtype=x.dtype)

        N_OUT = B * OH * OW

        grid = lambda meta: (
            triton.cdiv(N_OUT, meta['BLOCK_N']),
            triton.cdiv(OC, meta['BLOCK_OC']),
        )

        conv_relu_hardswish_nhwc_kernel[grid](
            x_nhwc, w, b, y_nhwc,
            B, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            N_OUT,
        )

        # convert back to NCHW
        y = y_nhwc.permute(0, 3, 1, 2).contiguous()
        return y