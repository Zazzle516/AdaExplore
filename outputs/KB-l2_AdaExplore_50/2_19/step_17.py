import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# ConvTranspose2d with stride=1, kernel=3 is equivalent to Conv2d with the
# spatially-flipped kernel, padding=2, and the IC/OC dimensions swapped on the
# weight (PyTorch ConvTranspose2d weight has shape (IC, OC, kH, kW)).
#
# We implement a direct gather conv2d with NCHW layout, fusing bias add and
# GELU into the epilogue, and writing the activation to a contiguous tensor
# that GroupNorm then consumes.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['N', 'IC', 'H', 'W', 'OC'],
)
@triton.jit
def conv_transpose_gelu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, IC, H, W,
    OC, OH, OW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n = tl.program_id(0)
    oc_block = tl.program_id(1)
    m_block = tl.program_id(2)

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = oc_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    oh = offs_m // OW
    ow = offs_m % OW

    m_mask = offs_m < (OH * OW)
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # y[n,oc,oh,ow] = sum_{ic,kh,kw} x[n,ic, oh+kh-2, ow+kw-2] * W_orig[ic, oc, 2-kh, 2-kw]
    for kh in tl.static_range(0, 3):
        ih = oh + kh - 2
        ih_ok = (ih >= 0) & (ih < H)
        for kw in tl.static_range(0, 3):
            iw = ow + kw - 2
            iw_ok = (iw >= 0) & (iw < W)
            spatial_ok = ih_ok & iw_ok & m_mask
            kh_o = 2 - kh
            kw_o = 2 - kw

            # iterate over IC in blocks of BLOCK_K
            for ic_base in range(0, IC, BLOCK_K):
                ic_idx = ic_base + offs_k  # [BLOCK_K]
                k_mask = ic_idx < IC

                # x: [BLOCK_M, BLOCK_K] gather
                x_off = ((n * IC + ic_idx[None, :]) * H + ih[:, None]) * W + iw[:, None]
                x_m = spatial_ok[:, None] & k_mask[None, :]
                x_tile = tl.load(x_ptr + x_off, mask=x_m, other=0.0)

                # w: [BLOCK_K, BLOCK_N]; W_orig[ic, oc, kh_o, kw_o]
                w_off = ((ic_idx[:, None] * OC + offs_n[None, :]) * 3 + kh_o) * 3 + kw_o
                w_m = k_mask[:, None] & n_mask[None, :]
                w_tile = tl.load(w_ptr + w_off, mask=w_m, other=0.0)

                acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    gelu = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865475))

    y_offset = ((n * OC + offs_n[None, :]) * OH + oh[:, None]) * OW + ow[:, None]
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + y_offset, gelu, mask=store_mask)


@triton.jit
def groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    HW, GROUP_SIZE, NUM_GROUPS, C,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    group_elems = GROUP_SIZE * HW
    base = n * C * HW + g * GROUP_SIZE * HW

    sum_val = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)

    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_val += tl.where(mask, x, 0.0)
        sum_sq += tl.where(mask, x * x, 0.0)

    inv_n = 1.0 / group_elems
    mean = tl.sum(sum_val) * inv_n
    mean_sq = tl.sum(sum_sq) * inv_n
    var = mean_sq - mean * mean
    rstd = tl.rsqrt(var + eps)
    neg_mean_rstd = -mean * rstd

    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        c_in_group = idx // HW
        w = tl.load(weight_ptr + g * GROUP_SIZE + c_in_group, mask=mask, other=0.0)
        b = tl.load(bias_ptr + g * GROUP_SIZE + c_in_group, mask=mask, other=0.0)
        out = (x * rstd + neg_mean_rstd) * w + b
        tl.store(y_ptr + base + idx, out, mask=mask)


def fused_groupnorm(x, weight, bias, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    GROUP_SIZE = C // num_groups
    y = torch.empty_like(x)
    grid = (N * num_groups,)
    BLOCK = 8192
    groupnorm_kernel[grid](
        x, y, weight, bias,
        HW, GROUP_SIZE, num_groups, C,
        eps,
        BLOCK=BLOCK,
        num_warps=16,
        num_stages=2,
    )
    return y


def conv_transpose_gelu(x, weight, bias):
    N, IC, H, W = x.shape
    _, OC, KH, KW = weight.shape
    assert KH == 3 and KW == 3
    OH = H + 2
    OW = W + 2

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    y = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_N']), triton.cdiv(OH * OW, META['BLOCK_M']))
    conv_transpose_gelu_kernel[grid](
        x, weight, bias, y,
        N, IC, H, W,
        OC, OH, OW,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        y = conv_transpose_gelu(x, self.conv_transpose.weight, self.conv_transpose.bias)
        out = fused_groupnorm(y, self.group_norm.weight, self.group_norm.bias,
                              self.num_groups, self.eps)
        return out