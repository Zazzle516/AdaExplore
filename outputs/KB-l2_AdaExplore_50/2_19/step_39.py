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

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64,  'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_OC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 64,  'BLOCK_OC': 32}, num_warps=4, num_stages=3),
    ],
    key=['N', 'IC', 'H', 'W', 'OC', 'OH', 'OW'],
)
@triton.jit
def _conv_bias_gelu_nhwc_kernel(
    x_ptr,        # input NHWC  [N, H, W, IC]
    w_ptr,        # weight [IC, OC, KH, KW]  (ConvTranspose2d weight)
    b_ptr,        # bias   [OC]
    y_ptr,        # output NHWC [N, OH, OW, OC]
    N, IC: tl.constexpr, H, W,
    OC: tl.constexpr, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n   = tl.program_id(0)
    pid_hw  = tl.program_id(1)
    pid_oc  = tl.program_id(2)

    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    ic_offs = tl.arange(0, IC)

    hw_mask = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs %  OW

    bias_vals = tl.load(b_ptr + oc_offs).to(tl.float32)
    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32) + bias_vals[None, :]

    base_n_x = pid_n * H * W * IC

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh - PAD
            iw = ow + kw - PAD
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & hw_mask

            # NHWC input: contiguous in IC
            x_off = base_n_x + ih[:, None] * (W * IC) + iw[:, None] * IC + ic_offs[None, :]
            x_vals = tl.load(x_ptr + x_off, mask=in_bounds[:, None], other=0.0)

            # flipped kernel for transposed conv
            fkh = (KH - 1) - kh
            fkw = (KW - 1) - kw
            w_off = ic_offs[:, None] * (OC * KH * KW) + oc_offs[None, :] * (KH * KW) + fkh * KW + fkw
            w_vals = tl.load(w_ptr + w_off)

            acc += tl.dot(x_vals, w_vals)

    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    out_off = pid_n * (OH * OW * OC) + hw_offs[:, None] * OC + oc_offs[None, :]
    out_mask = hw_mask[:, None]
    tl.store(y_ptr + out_off, gelu, mask=out_mask)


def conv_bias_gelu_nhwc(x_nhwc, weight, bias):
    """
    x_nhwc: [N, H, W, IC] fp32 contiguous (NHWC)
    weight: [IC, OC, KH, KW] (ConvTranspose2d weight)
    bias:   [OC]
    Returns: y NHWC fp32 [N, OH, OW, OC]
    """
    N, H, W, IC = x_nhwc.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w
    OH = H + KH - 1
    OW = W + KW - 1
    PAD = KH - 1

    y = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)

    grid = lambda META: (N, triton.cdiv(OH * OW, META['BLOCK_HW']), triton.cdiv(OC, META['BLOCK_OC']))
    _conv_bias_gelu_nhwc_kernel[grid](
        x_nhwc, weight, bias, y,
        N, IC, H, W,
        OC, OH, OW,
        KH=KH, KW=KW,
        PAD=PAD,
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

    base_n = n * H * W * C
    c_start = g * GROUP_SIZE
    c_offs = c_start + tl.arange(0, GROUP_SIZE)  # [GROUP_SIZE]

    # Single-pass sum & sum_sq accumulation
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq  = tl.zeros((), dtype=tl.float32)

    for hw_start in tl.range(0, HW, BLOCK):
        hw_idx = hw_start + tl.arange(0, BLOCK)
        hw_mask = hw_idx < HW
        offs = base_n + hw_idx[:, None] * C + c_offs[None, :]
        v = tl.load(x_ptr + offs, mask=hw_mask[:, None], other=0.0).to(tl.float32)
        sum_val += tl.sum(v)
        sum_sq  += tl.sum(v * v)

    mean = sum_val * inv_n
    var = sum_sq * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    w_vals = tl.load(weight_ptr + c_offs).to(tl.float32)
    b_vals = tl.load(bias_ptr   + c_offs).to(tl.float32)

    scale = w_vals * rstd
    shift = b_vals - mean * scale

    base_out_n = n * C * HW
    for hw_start in tl.range(0, HW, BLOCK):
        hw_idx = hw_start + tl.arange(0, BLOCK)
        hw_mask = hw_idx < HW
        offs = base_n + hw_idx[:, None] * C + c_offs[None, :]
        v = tl.load(x_ptr + offs, mask=hw_mask[:, None], other=0.0).to(tl.float32)
        out = v * scale[None, :] + shift[None, :]

        out_offs = base_out_n + c_offs[None, :] * HW + hw_idx[:, None]
        tl.store(out_ptr + out_offs, out, mask=hw_mask[:, None])


def groupnorm_nhwc_to_nchw(x_nhwc, weight, bias, num_groups, eps=1e-5):
    N, H, W, C = x_nhwc.shape
    HW = H * W
    GROUP_SIZE = C // num_groups
    out = torch.empty((N, C, H, W), device=x_nhwc.device, dtype=x_nhwc.dtype)

    grid = (N * num_groups,)
    BLOCK = 1024
    _groupnorm_nhwc_to_nchw_kernel[grid](
        x_nhwc, out, weight, bias,
        N, C, H, W,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=GROUP_SIZE,
        HW=HW,
        eps=eps,
        BLOCK=BLOCK,
        num_warps=8,
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
        # conv_transpose2d with stride=1, padding=0: implement via direct conv2d-like kernel
        if self.stride == 1 and self.conv_transpose.padding == (0, 0) and self.conv_transpose.output_padding == (0, 0):
            # NCHW -> NHWC
            x_nhwc = x.permute(0, 2, 3, 1).contiguous()
            y_nhwc = conv_bias_gelu_nhwc(
                x_nhwc,
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