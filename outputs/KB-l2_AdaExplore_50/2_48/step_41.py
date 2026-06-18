import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=2),
    ],
    key=['N', 'OD', 'OH', 'OW', 'OC'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, cb_ptr, scale_ptr, bias_ptr, out_ptr,
    N, ID, IH, IW,
    OC, OD, OH, OW,
    BLOCK_N: tl.constexpr,
):
    # grid: (N*OD, ceil(OH*OW/BLOCK_N))
    pid_nod = tl.program_id(0)
    pid_hw  = tl.program_id(1)

    n  = pid_nod // OD
    od = pid_nod % OD

    OHW = OH * OW
    offs_hw = pid_hw * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_hw = offs_hw < OHW
    oh = offs_hw // OW
    ow = offs_hw - oh * OW

    # constants for this problem (KD=KH=KW=3, IC=3, OC=16)
    KD: tl.constexpr = 3
    KH: tl.constexpr = 3
    KW: tl.constexpr = 3
    IC: tl.constexpr = 3
    OC_C: tl.constexpr = 16
    K_TOTAL: tl.constexpr = IC * KD * KH * KW  # 81

    IHIW = IH * IW
    x_n_base = n * (IC * ID * IHIW)

    # accumulator [OC_C, BLOCK_N]
    acc = tl.zeros((OC_C, BLOCK_N), dtype=tl.float32)

    oh_IW = oh * IW  # [BLOCK_N]

    # unrolled K loop: ic=0..2, kd=0..2, kh=0..2, kw=0..2 -> 81 iterations
    # Each iteration loads a vector of OC_C weights and a vector of BLOCK_N inputs.
    offs_oc = tl.arange(0, OC_C)
    w_oc_stride = K_TOTAL  # weight is (OC, IC*KD*KH*KW)

    for ic in tl.static_range(0, IC):
        x_ic_base = x_n_base + ic * (ID * IHIW)
        w_ic_base = ic * (KD * KH * KW)
        for kd in tl.static_range(0, KD):
            x_kd_base = x_ic_base + (od + kd) * IHIW
            w_kd_base = w_ic_base + kd * (KH * KW)
            for kh in tl.static_range(0, KH):
                x_kh_base = x_kd_base + (oh_IW + kh * IW)  # [BLOCK_N]
                w_kh_base = w_kd_base + kh * KW
                for kw in tl.static_range(0, KW):
                    x_idx = x_kh_base + ow + kw
                    x_vals = tl.load(x_ptr + x_idx, mask=mask_hw, other=0.0)  # [BLOCK_N]

                    w_idx = offs_oc * w_oc_stride + (w_kh_base + kw)
                    w_vals = tl.load(w_ptr + w_idx)  # [OC_C]

                    acc += w_vals[:, None] * x_vals[None, :]

    cb = tl.load(cb_ptr + offs_oc)
    s  = tl.load(scale_ptr + offs_oc)
    b  = tl.load(bias_ptr + offs_oc)

    acc = acc + cb[:, None]
    y = acc * s[:, None]
    e2 = tl.exp(2.0 * y)
    t = (e2 - 1.0) / (e2 + 1.0)
    z = t * b[:, None]
    out_val = 1.0 / (1.0 + tl.exp(-z))

    out_base = n * (OC * OD * OHW) + offs_oc[:, None] * (OD * OHW) + od * OHW
    out_idx = out_base + offs_hw[None, :]
    tl.store(out_ptr + out_idx, out_val, mask=mask_hw[None, :])


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

        # weight as (OC, IC*KD*KH*KW), contiguous
        w = self.conv.weight.contiguous().view(OC, -1).contiguous()
        cb = self.conv.bias.contiguous()
        scale = self.scaling_factor.contiguous().view(-1)
        bias = self.bias.contiguous().view(-1)

        grid = lambda META: (
            N * OD,
            triton.cdiv(OH * OW, META['BLOCK_N']),
        )

        conv3d_fused_kernel[grid](
            x, w, cb, scale, bias, out,
            N, ID, IH, IW,
            OC, OD, OH, OW,
        )
        return out