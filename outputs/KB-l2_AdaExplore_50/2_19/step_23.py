import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# ---------------------------------------------------------------------------
# Conv2d + bias + GELU kernel (replaces ConvTranspose2d with stride=1, k=3, p=0)
# Equivalent to: pad input by 2 (k-1), then conv2d with flipped kernel.
# Output layout: NHWC for fast GroupNorm.
# ---------------------------------------------------------------------------

@triton.jit
def _conv_bias_gelu_nhwc_kernel(
    x_ptr,        # input  [N, IC, H, W]  (contiguous NCHW)
    w_ptr,        # weight [IC, OC, KH, KW]  (ConvTranspose2d weight, OC = out_channels)
    b_ptr,        # bias   [OC]
    y_ptr,        # output [N, OH, OW, OC]  NHWC
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # Program IDs
    pid_n   = tl.program_id(0)               # batch
    pid_hw  = tl.program_id(1)               # tile over OH*OW
    pid_oc  = tl.program_id(2)               # tile over OC

    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]

    hw_mask = hw_offs < (OH * OW)
    oc_mask = oc_offs < OC

    oh = hw_offs // OW
    ow = hw_offs %  OW

    # Initialize accumulator with bias
    bias_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32) + bias_vals[None, :]

    # ConvTranspose2d (stride=1, padding=0, kernel=KH×KW) is equivalent to
    # conv2d on input padded by (KH-1, KW-1) with kernel flipped (kh, kw) -> (KH-1-kh, KW-1-kw)
    # Loop over IC and kernel
    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                # input spatial pos in original (unpadded) coords:
                # padded pos = oh + kh, original ih = oh + kh - PAD
                ih = oh + kh - PAD
                iw = ow + kw - PAD
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & hw_mask

                x_off = pid_n * (IC * H * W) + ic * (H * W) + ih * W + iw
                x_vals = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0).to(tl.float32)  # [BLOCK_HW]

                # weight: ConvTranspose2d weight [IC, OC, KH, KW]
                # use flipped kernel: (KH-1-kh, KW-1-kw)
                fkh = (KH - 1) - kh
                fkw = (KW - 1) - kw
                w_off = ic * (OC * KH * KW) + oc_offs * (KH * KW) + fkh * KW + fkw
                w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0).to(tl.float32)  # [BLOCK_OC]

                acc += x_vals[:, None] * w_vals[None, :]

    # GELU (exact)
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Store as NHWC: y[n, oh, ow, oc] = y_ptr[n*OH*OW*OC + hw*OC + oc]
    out_off = pid_n * (OH * OW * OC) + hw_offs[:, None] * OC + oc_offs[None, :]
    out_mask = hw_mask[:, None] & oc_mask[None, :]
    tl.store(y_ptr + out_off, gelu, mask=out_mask)


def conv_bias_gelu_nhwc(x, weight, bias):
    """
    x:      [N, IC, H, W]  fp32 contiguous
    weight: [IC, OC, KH, KW] (ConvTranspose2d weight)
    bias:   [OC]
    Returns: y NHWC fp32 [N, OH, OW, OC] where OH=H+KH-1, OW=W+KW-1
    """
    N, IC, H, W = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w
    OH = H + KH - 1
    OW = W + KW - 1
    PAD = KH - 1  # assume KH==KW

    y = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

    BLOCK_HW = 64
    BLOCK_OC = 64

    grid = (N, triton.cdiv(OH * OW, BLOCK_HW), triton.cdiv(OC, BLOCK_OC))
    _conv_bias_gelu_nhwc_kernel[grid](
        x, weight, bias, y,
        N, IC, H, W,
        OC, OH, OW,
        KH=KH, KW=KW,
        PAD=PAD,
        BLOCK_HW=BLOCK_HW,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )
    return y


# ---------------------------------------------------------------------------
# GroupNorm on NHWC layout
# Input:  y [N, H, W, C]  contiguous
# Output: z [N, C, H, W]  contiguous (back to NCHW for downstream compatibility)
# ---------------------------------------------------------------------------

@triton.jit
def _groupnorm_nhwc_to_nchw_kernel(
    x_ptr,         # NHWC input
    out_ptr,       # NCHW output
    weight_ptr,    # [C]
    bias_ptr,      # [C]
    N, C, H, W,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,   # C // NUM_GROUPS
    HW,
    eps,
    BLOCK: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    group_elems = GROUP_SIZE * HW
    inv_n = 1.0 / group_elems

    # Pass 1: accumulate sum and sum of squares
    # In NHWC, for fixed (n, g), the channels in this group are
    # c in [g*GROUP_SIZE, (g+1)*GROUP_SIZE). For each spatial pos hw, those
    # GROUP_SIZE channels are contiguous in memory.
    # We tile by BLOCK over hw, load BLOCK x GROUP_SIZE block.
    sum_val = 0.0
    sum_sq = 0.0

    base_n = n * H * W * C
    c_start = g * GROUP_SIZE
    c_offs = c_start + tl.arange(0, GROUP_SIZE)  # [GROUP_SIZE]

    for hw_start in tl.range(0, HW, BLOCK):
        hw_idx = hw_start + tl.arange(0, BLOCK)
        hw_mask = hw_idx < HW
        # offsets: base_n + hw_idx*C + c_offs
        offs = base_n + hw_idx[:, None] * C + c_offs[None, :]
        mask = hw_mask[:, None]
        v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(v)
        sum_sq += tl.sum(v * v)

    mean = sum_val * inv_n
    var = sum_sq * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Load affine for these channels
    w_vals = tl.load(weight_ptr + c_offs).to(tl.float32)  # [GROUP_SIZE]
    b_vals = tl.load(bias_ptr   + c_offs).to(tl.float32)  # [GROUP_SIZE]

    # Pass 2: normalize and write to NCHW output
    # NCHW offset: n*C*HW + c*HW + hw
    base_out_n = n * C * HW
    for hw_start in tl.range(0, HW, BLOCK):
        hw_idx = hw_start + tl.arange(0, BLOCK)
        hw_mask = hw_idx < HW
        offs = base_n + hw_idx[:, None] * C + c_offs[None, :]
        mask = hw_mask[:, None]
        v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        out = (v - mean) * rstd * w_vals[None, :] + b_vals[None, :]

        # Write NCHW
        out_offs = base_out_n + c_offs[None, :] * HW + hw_idx[:, None]
        tl.store(out_ptr + out_offs, out, mask=mask)


def groupnorm_nhwc_to_nchw(x_nhwc, weight, bias, num_groups, eps=1e-5):
    N, H, W, C = x_nhwc.shape
    HW = H * W
    GROUP_SIZE = C // num_groups
    out = torch.empty((N, C, H, W), device=x_nhwc.device, dtype=x_nhwc.dtype)

    grid = (N * num_groups,)
    BLOCK = 256
    _groupnorm_nhwc_to_nchw_kernel[grid](
        x_nhwc, out, weight, bias,
        N, C, H, W,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=GROUP_SIZE,
        HW=HW,
        eps=eps,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = x.contiguous()
        # conv_transpose2d with stride=1, padding=0: implement via direct conv2d-like kernel
        if self.stride == 1 and self.conv_transpose.padding == (0, 0) and self.conv_transpose.output_padding == (0, 0):
            y_nhwc = conv_bias_gelu_nhwc(
                x,
                self.conv_transpose.weight,
                self.conv_transpose.bias,
            )
            out = groupnorm_nhwc_to_nchw(
                y_nhwc,
                self.group_norm.weight,
                self.group_norm.bias,
                self.num_groups,
                self.group_norm.eps,
            )
            return out
        else:
            # Fallback path
            x = self.conv_transpose(x)
            x = F.gelu(x)
            x = self.group_norm(x)
            return x