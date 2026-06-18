import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_W': 32}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C', 'D', 'H', 'W'],
)
@triton.jit
def fused_pool_lse_relu_kernel(
    in_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, od, oh, ow_block); processes BLOCK_W outputs along W
    pid = tl.program_id(0)
    OW_B = OW // BLOCK_W
    owb = pid % OW_B
    tmp = pid // OW_B
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    d0 = od * 2
    h0 = oh * 2
    w0_base = owb * BLOCK_W * 2

    offs_c = tl.arange(0, BLOCK_C)  # [BLOCK_C]
    # Load 2*BLOCK_W contiguous w-values per (d,h) corner
    offs_w2 = tl.arange(0, BLOCK_W * 2)  # [2*BLOCK_W]
    w_in = w0_base + offs_w2  # [2*BLOCK_W]

    base = n * stride_n + offs_c[None, :] * stride_c + w_in[:, None] * stride_w
    # 4 loads, one per (d,h) corner
    v00 = tl.load(in_ptr + base + (d0 + 0) * stride_d + (h0 + 0) * stride_h)
    v01 = tl.load(in_ptr + base + (d0 + 0) * stride_d + (h0 + 1) * stride_h)
    v10 = tl.load(in_ptr + base + (d0 + 1) * stride_d + (h0 + 0) * stride_h)
    v11 = tl.load(in_ptr + base + (d0 + 1) * stride_d + (h0 + 1) * stride_h)

    # max across (d,h)
    m_dh = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))  # [2*BLOCK_W, BLOCK_C]

    # reshape to [BLOCK_W, 2, BLOCK_C] and reduce w-pair
    m_dh = tl.reshape(m_dh, (BLOCK_W, 2, BLOCK_C))
    pooled = tl.max(m_dh, axis=1)  # [BLOCK_W, BLOCK_C]

    # logsumexp across channels (no mask: C == BLOCK_C)
    max_val = tl.max(pooled, axis=1)  # [BLOCK_W]
    shifted = pooled - max_val[:, None]
    exp_vals = tl.exp(shifted)
    sum_exp = tl.sum(exp_vals, axis=1)
    lse = max_val + tl.log(sum_exp)
    res = tl.maximum(lse, 0.0)  # [BLOCK_W]

    offs_w = tl.arange(0, BLOCK_W)
    out_w = owb * BLOCK_W + offs_w
    out_offset = n * out_stride_n + od * out_stride_d + oh * out_stride_h + out_w * out_stride_w
    tl.store(out_ptr + out_offset, res)


def fused_pool_lse_relu(x):
    # x: [N, C, D, H, W] -> output [N, 1, D//2, H//2, W//2]
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    N, C, D, H, W = x.shape
    OD, OH, OW = D // 2, H // 2, W // 2

    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C < 16:
        BLOCK_C = 16

    sN, sC, sD, sH, sW = x.stride()
    o_sN = OD * OH * OW
    o_sD = OH * OW
    o_sH = OW
    o_sW = 1

    grid = lambda META: (N * OD * OH * (OW // META['BLOCK_W']),)

    fused_pool_lse_relu_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        sN, sC, sD, sH, sW,
        o_sN, o_sD, o_sH, o_sW,
        BLOCK_C=BLOCK_C,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = fused_pool_lse_relu(x)
        return x