import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# ============================================================
# Custom ConvTranspose2d kernel for stride=1, k=3
# ConvT with stride=1 is equivalent to conv2d with full padding=k-1
# Input:  (N, IC, H, W)
# Output: (N, OC, H+2, W+2)  for k=3, stride=1
# weight: (IC, OC, K, K)  (PyTorch ConvTranspose2d layout)
# bias:   (OC,)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64,  'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64,  'BLOCK_OC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'H', 'W'],
)
@triton.jit
def conv_transpose_gelu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, IC: tl.constexpr, OC: tl.constexpr,
    H, W, OH, OW,
    K: tl.constexpr, PAD: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr,
):
    # Grid: (batch, spatial_tiles, oc_tiles)
    n = tl.program_id(0)
    pid_sp = tl.program_id(1)
    pid_oc = tl.program_id(2)

    OHW = OH * OW
    sp_start = pid_sp * BLOCK_N
    sp_offs = sp_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    sp_mask = sp_offs < OHW

    oh = sp_offs // OW          # [BLOCK_N]
    ow = sp_offs % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    acc = tl.zeros([BLOCK_N, BLOCK_OC], dtype=tl.float32)

    # ConvT stride=1 with padding=0 is equivalent to:
    #   y[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, oh - kh + (K-1), ow - kw + (K-1)] * w[ic, oc, K-1-kh, K-1-kw]
    # Equivalently (set ih = oh - kh + (K-1)):
    #   for each (kh, kw): ih = oh + kh - PAD, iw = ow + kw - PAD, where PAD = K-1
    # and weight index is w[ic, oc, K-1-kh, K-1-kw]
    # But we can just iterate kh, kw straightforwardly using the conv2d equivalence:
    # y = conv2d(x, w_flipped_swapped, padding=K-1)
    # where w_flipped_swapped[oc, ic, kh, kw] = w[ic, oc, K-1-kh, K-1-kw]
    #
    # To avoid materializing flipped weight, do it inline:

    # Iterate over kh, kw, ic
    for kh in tl.static_range(0, K):
        ih = oh + kh - PAD          # [BLOCK_N]
        ih_valid = (ih >= 0) & (ih < H)
        for kw in tl.static_range(0, K):
            iw = ow + kw - PAD
            iw_valid = (iw >= 0) & (iw < W)
            spatial_valid = ih_valid & iw_valid & sp_mask

            # weight position (kh_w, kw_w) = (K-1-kh, K-1-kw) for flipped
            kh_w = K - 1 - kh
            kw_w = K - 1 - kw

            # Load input slice [BLOCK_N, IC] and weight slice [IC, BLOCK_OC]
            # x[n, ic, ih, iw] for ic in 0..IC
            # offset: n*IC*H*W + ic*H*W + ih*W + iw
            ic_range = tl.arange(0, IC)  # [IC]

            x_offs = (n * IC * H * W
                      + ic_range[None, :] * H * W
                      + ih[:, None] * W
                      + iw[:, None])  # [BLOCK_N, IC]
            x_mask = spatial_valid[:, None]
            x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)  # [BLOCK_N, IC]

            # weight: w[ic, oc, kh_w, kw_w]
            # layout: (IC, OC, K, K) -> ic*OC*K*K + oc*K*K + kh_w*K + kw_w
            w_offs = (ic_range[:, None] * OC * K * K
                      + oc_offs[None, :] * K * K
                      + kh_w * K + kw_w)  # [IC, BLOCK_OC]
            w_mask = oc_mask[None, :]
            w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [IC, BLOCK_OC]

            acc += tl.dot(x_vals, w_vals)

    # Add bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + b_vals[None, :]

    # GELU (exact, erf-based)
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * 0.7071067811865475))

    # Store: y[n, oc, oh, ow] = n*OC*OHW + oc*OHW + oh*OW + ow
    y_offs = (n * OC * OHW
              + oc_offs[None, :] * OHW
              + sp_offs[:, None])  # [BLOCK_N, BLOCK_OC]
    y_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(y_ptr + y_offs, gelu, mask=y_mask)


def conv_transpose_gelu(x, weight, bias):
    """
    x:      (N, IC, H, W)
    weight: (IC, OC, K, K)  -- PyTorch ConvTranspose2d weight layout
    bias:   (OC,)
    Returns: (N, OC, H+K-1, W+K-1) -- stride=1, padding=0
    """
    N, IC, H, W = x.shape
    IC_w, OC, K, _ = weight.shape
    assert IC == IC_w
    PAD = K - 1
    OH = H + 2 * PAD - (K - 1)  # = H + K - 1
    OW = W + 2 * PAD - (K - 1)
    # For stride=1 convT (no padding): output = H + K - 1
    OH = H + K - 1
    OW = W + K - 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    y = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    OHW = OH * OW
    grid = lambda meta: (
        N,
        triton.cdiv(OHW, meta['BLOCK_N']),
        triton.cdiv(OC, meta['BLOCK_OC']),
    )
    conv_transpose_gelu_kernel[grid](
        x, weight, bias, y,
        N, IC, OC,
        H, W, OH, OW,
        K, PAD,
    )
    return y


# ============================================================
# Fused GroupNorm kernel (input is already GELU-activated)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=2),
    ],
    key=['C', 'HW', 'GROUP_SIZE'],
)
@triton.jit
def groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, HW, GROUP_SIZE: tl.constexpr, NUM_GROUPS,
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
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += x
        sum_sq += x * x

    mean = tl.sum(sum_val) / group_elems
    mean_sq = tl.sum(sum_sq) / group_elems
    var = mean_sq - mean * mean
    rstd = tl.rsqrt(var + eps)

    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        c_in_group = idx // HW
        c_global = g * GROUP_SIZE + c_in_group
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0)
        out = (x - mean) * rstd * w + b
        tl.store(y_ptr + base + idx, out, mask=mask)


def fused_groupnorm(x, weight, bias, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    GROUP_SIZE = C // num_groups
    x = x.contiguous()
    y = torch.empty_like(x)
    grid = (N * num_groups,)
    groupnorm_kernel[grid](
        x, y, weight, bias,
        N, C, HW, GROUP_SIZE, num_groups,
        eps,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = 1e-5

    def forward(self, x):
        if self.stride == 1 and self.kernel_size == 3:
            x = x.contiguous()
            y = conv_transpose_gelu(x, self.conv_transpose.weight, self.conv_transpose.bias)
        else:
            y = self.conv_transpose(x)
            y = F.gelu(y)
        y = fused_groupnorm(y, self.group_norm.weight, self.group_norm.bias,
                            self.num_groups, self.eps)
        return y