import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
    ],
    key=['OC', 'OH', 'OW', 'K_TOTAL'],
)
@triton.jit
def conv_relu_hswish_nhwc_kernel(
    x_ptr,    # (N, IH, IW, IC)  NHWC
    w_ptr,    # (K_TOTAL, OC)    where K_TOTAL = KH*KW*IC, packed
    b_ptr,    # (OC,)
    out_ptr,  # (N, OH, OW, OC)  NHWC
    N, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr, IC: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_oc = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    mask_oc = offs_oc < OC
    mask_hw = offs_hw < (OH * OW)

    oh = offs_hw // OW
    ow = offs_hw % OW

    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    # Iterate over kernel positions; inner channel is unrolled via BLOCK_K = IC
    # Build K dim contiguously: for (kh, kw) load IC channels of x at (n, oh+kh, ow+kw, :)
    # and weight slice w[kh*KW*IC + kw*IC : ... , :]
    offs_ic = tl.arange(0, IC)

    x_n_off = pid_n * (IH * IW * IC)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # (BLOCK_HW,)
            iw = ow + kw  # (BLOCK_HW,)
            # x offsets: (BLOCK_HW, IC)
            x_off = x_n_off + ih[:, None] * (IW * IC) + iw[:, None] * IC + offs_ic[None, :]
            x_mask = mask_hw[:, None]
            x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # (BLOCK_HW, IC)

            # weight: (IC, BLOCK_OC) from row (kh*KW + kw)*IC ... (+IC)
            k_base = (kh * KW + kw) * IC
            w_rows = k_base + offs_ic  # (IC,)
            w_off = w_rows[:, None] * OC + offs_oc[None, :]
            w_mask = mask_oc[None, :]
            w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # (IC, BLOCK_OC)

            acc += tl.dot(x_vals, w_vals, out_dtype=tl.float32)

    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)  # (BLOCK_OC,)
    acc = acc + b_vals[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)
    # HardSwish: x * clamp((x+3)/6, 0, 1)
    hs = tl.minimum(tl.maximum((acc + 3.0) * (1.0 / 6.0), 0.0), 1.0)
    acc = acc * hs

    # Store NHWC: out[n, oh, ow, oc]
    out_n_off = pid_n * (OH * OW * OC)
    out_off = out_n_off + offs_hw[:, None] * OC + offs_oc[None, :]
    out_mask = mask_hw[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-pack weight to NHWC-friendly (KH*KW*IC, OC)
        with torch.no_grad():
            w = self.conv.weight.detach()  # (OC, IC, KH, KW)
            # Want layout: row = (kh*KW + kw)*IC + ic, col = oc
            # Permute to (KH, KW, IC, OC) -> reshape (K_TOTAL, OC)
            w_packed = w.permute(2, 3, 1, 0).contiguous()  # (KH, KW, IC, OC)
            OC = w.shape[0]
            KH = w.shape[2]
            KW = w.shape[3]
            IC = w.shape[1]
            w_packed = w_packed.view(KH * KW * IC, OC).contiguous()
            self.register_buffer('w_packed', w_packed)
            self.register_buffer('bias_buf', self.conv.bias.detach().contiguous())
        self._KH = self.conv.kernel_size[0]
        self._KW = self.conv.kernel_size[1]

    def forward(self, x):
        x = x.cuda()
        # NCHW -> NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        N, IH, IW, IC = x_nhwc.shape
        OC = self.out_channels
        KH = self._KH
        KW = self._KW
        OH = IH - KH + 1
        OW = IW - KW + 1
        K_TOTAL = KH * KW * IC

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        w_packed = self.w_packed
        b = self.bias_buf
        if w_packed.device != x.device:
            w_packed = w_packed.to(x.device)
            b = b.to(x.device)

        grid = lambda meta: (
            N,
            triton.cdiv(OH * OW, meta['BLOCK_HW']),
            triton.cdiv(OC, meta['BLOCK_OC']),
        )

        conv_relu_hswish_nhwc_kernel[grid](
            x_nhwc, w_packed, b, out_nhwc,
            N, IH, IW,
            OC, OH, OW,
            KH, KW, IC,
            K_TOTAL,
        )

        # NHWC -> NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out