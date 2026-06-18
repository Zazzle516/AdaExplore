import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512, 'BLOCK_OC': 16}, num_warps=4, num_stages=2),
    ],
    key=['N', 'IC', 'ID', 'IH', 'IW', 'OC', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, cb_ptr, scale_ptr, bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    BLOCK_N: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)   # over (N * OD)
    pid_hw = tl.program_id(1)  # over output spatial blocks (OH*OW / BLOCK_N)
    pid_oc = tl.program_id(2)  # over OC blocks

    n = pid_n // OD
    od = pid_n % OD

    offs_hw = pid_hw * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_hw = offs_hw < (OH * OW)
    oh = offs_hw // OW
    ow = offs_hw % OW

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # accumulator [BLOCK_OC, BLOCK_N]
    acc = tl.zeros((BLOCK_OC, BLOCK_N), dtype=tl.float32)

    # input base for this (n, od)
    x_n_base = n * (IC * ID * IH * IW)

    # weight layout: (OC, IC, KD, KH, KW)
    w_oc_stride = IC * KD * KH * KW

    IHIW = IH * IW
    ow_base = ow  # [BLOCK_N]
    oh_IW = oh * IW  # [BLOCK_N]

    for ic in tl.static_range(0, 3):  # IC = 3
        x_ic_base = x_n_base + ic * (ID * IHIW)
        w_ic_base = ic * (KD * KH * KW)
        for kd in tl.static_range(0, 3):  # KD = 3
            id_in_off = (od + kd) * IHIW
            x_kd_base = x_ic_base + id_in_off
            w_kd_base = w_ic_base + kd * (KH * KW)
            for kh in tl.static_range(0, 3):  # KH = 3
                ih_off = oh_IW + kh * IW  # [BLOCK_N]
                x_kh_base = x_kd_base + ih_off
                w_kh_base = w_kd_base + kh * KW
                for kw in tl.static_range(0, 3):  # KW = 3
                    x_idx = x_kh_base + ow_base + kw
                    x_vals = tl.load(x_ptr + x_idx, mask=mask_hw, other=0.0)  # [BLOCK_N]

                    w_idx = offs_oc * w_oc_stride + w_kh_base + kw
                    w_vals = tl.load(w_ptr + w_idx, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                    acc += w_vals[:, None] * x_vals[None, :]

    # add conv bias
    cb = tl.load(cb_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += cb[:, None]

    # epilogue: * scale, tanh, * bias, sigmoid
    s = tl.load(scale_ptr + offs_oc, mask=mask_oc, other=0.0)
    b = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)

    y = acc * s[:, None]
    e2 = tl.exp(2.0 * y)
    t = (e2 - 1.0) / (e2 + 1.0)
    z = t * b[:, None]
    out_val = 1.0 / (1.0 + tl.exp(-z))

    # store: out shape (N, OC, OD, OH, OW)
    out_base = n * (OC * OD * OH * OW) + offs_oc[:, None] * (OD * OH * OW) + od * (OH * OW)
    out_idx = out_base + offs_hw[None, :]
    mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_idx, out_val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        w = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()
        scale = self.scaling_factor.contiguous().view(-1)
        bias = self.bias.contiguous().view(-1)

        grid = lambda META: (
            N * OD,
            triton.cdiv(OH * OW, META['BLOCK_N']),
            triton.cdiv(OC, META['BLOCK_OC']),
        )

        conv3d_fused_kernel[grid](
            x, w, cb, scale, bias, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
        )
        return out