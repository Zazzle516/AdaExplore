import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=3),
    ],
    key=['OD', 'OH', 'OW', 'IC_C', 'OC_C'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, scale_ptr, bias2_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr, OC_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)  # tile of output spatial positions
    pid_n = tl.program_id(1)  # batch index

    # output spatial linear index range
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < (OD * OH * OW)

    # decompose linear m into (od, oh, ow)
    od = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    n = pid_n

    # accumulator: [BLOCK_M, OC_C]
    acc = tl.zeros((BLOCK_M, OC_C), dtype=tl.float32)

    oc_offs = tl.arange(0, OC_C)  # [OC_C]

    # Preload entire weight tensor [OC_C, IC_C * KD * KH * KW] into registers
    K_TOT: tl.constexpr = IC_C * KD * KH * KW
    k_offs = tl.arange(0, K_TOT)  # [K_TOT]
    # weight layout [OC, IC, KD, KH, KW] flattened in last dim
    w_all = tl.load(w_ptr + oc_offs[:, None] * K_TOT + k_offs[None, :])  # [OC_C, K_TOT]

    # base input offset for batch n
    base_n = n * IC_C * ID * IH * IW

    for ic in tl.static_range(0, IC_C):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    in_off = base_n + ic * (ID * IH * IW) + (od + kd) * (IH * IW) + (oh + kh) * IW + (ow + kw)
                    x_vals = tl.load(x_ptr + in_off, mask=m_mask, other=0.0)  # [BLOCK_M]

                    k_idx: tl.constexpr = ((ic * KD + kd) * KH + kh) * KW + kw
                    w_vals = w_all[:, k_idx]  # [OC_C]

                    acc += x_vals[:, None] * w_vals[None, :]

    # bias
    b_vals = tl.load(b_ptr + oc_offs)  # [OC_C]
    acc = acc + b_vals[None, :]

    # scaling_factor (per OC)
    s_vals = tl.load(scale_ptr + oc_offs)  # [OC_C]
    y = acc * s_vals[None, :]

    # tanh
    e2 = tl.exp(2.0 * y)
    t = (e2 - 1.0) / (e2 + 1.0)

    # bias2
    b2_vals = tl.load(bias2_ptr + oc_offs)  # [OC_C]
    z = t * b2_vals[None, :]

    # sigmoid
    out = 1.0 / (1.0 + tl.exp(-z))

    # store: output layout [N, OC, OD, OH, OW]
    # out_off = ((n*OC + oc)*OD + od)*OH*OW + oh*OW + ow
    out_off = ((n * OC_C + oc_offs[None, :]) * OD + od[:, None]) * (OH * OW) + oh[:, None] * OW + ow[:, None]
    out_mask = m_mask[:, None] & (oc_offs[None, :] < OC_C)
    tl.store(out_ptr + out_off, out, mask=out_mask)


def fused_conv3d(x, weight, bias, scale, bias2):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    scale = scale.contiguous().view(-1)
    bias2 = bias2.contiguous().view(-1)

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    OC_C = OC  # 16
    IC_C = IC  # 3

    grid = lambda META: (triton.cdiv(OD * OH * OW, META['BLOCK_M']), N)

    conv3d_fused_kernel[grid](
        x, weight, bias, scale, bias2, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        IC_C, OC_C,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return fused_conv3d(
            x, self.conv.weight, self.conv.bias,
            self.scaling_factor, self.bias,
        )