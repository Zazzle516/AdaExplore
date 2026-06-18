import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 256}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_relu_hswish_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oh = offs_hw // OW
    ow = offs_hw % OW

    mask_oc = offs_oc < OC
    mask_hw = offs_hw < (OH * OW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # GEMM-style: K = IC * KH * KW, with implicit im2col
    # We tile over K with BLOCK_K = IC (since IC=8 small, can be entire IC at once)
    # Loop kh, kw outer, IC vectorized as K dim into tl.dot
    # A: (BLOCK_OC, IC) - weight slice per (kh,kw)
    # B: (IC, BLOCK_HW) - input slice per (kh,kw)
    BLOCK_K: tl.constexpr = IC

    ic_range = tl.arange(0, BLOCK_K)

    x_base = pid_n * (IC * IH * IW)
    w_base = 0

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # (BLOCK_HW,)
            iw = ow + kw  # (BLOCK_HW,)
            # B: (IC, BLOCK_HW)
            x_off = x_base + ic_range[:, None] * (IH * IW) + ih[None, :] * IW + iw[None, :]
            x_mask = mask_hw[None, :]
            b_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

            # A: (BLOCK_OC, IC)
            w_off = offs_oc[:, None] * (IC * KH * KW) + ic_range[None, :] * (KH * KW) + kh * KW + kw
            w_mask = mask_oc[:, None]
            a_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

            acc += tl.dot(a_tile, b_tile, allow_tf32=True)

    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b_vals[:, None]

    # ReLU
    acc = tl.maximum(acc, 0.0)
    # HardSwish
    hs = tl.minimum(tl.maximum((acc + 3.0) * (1.0 / 6.0), 0.0), 1.0)
    acc = acc * hs

    out_off = (pid_n * OC * OH * OW) + offs_oc[:, None] * (OH * OW) + offs_hw[None, :]
    mask_out = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask_out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = w.shape[0]
        KH = w.shape[2]
        KW = w.shape[3]
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']), triton.cdiv(OH * OW, meta['BLOCK_HW']))

        conv_relu_hswish_gemm_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
        )
        return out