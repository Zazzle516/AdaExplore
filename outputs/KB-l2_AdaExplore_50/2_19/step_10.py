import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def gelu_groupnorm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    N, C, HW, G, CPG,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    base = n * C * HW + g * CPG * HW

    sum_val = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)

    num_iters: tl.constexpr = (GROUP_SIZE + BLOCK - 1) // BLOCK

    for i in tl.static_range(num_iters):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < GROUP_SIZE
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
        gelu = tl.where(mask, gelu, 0.0)
        sum_val += gelu
        sum_sq += gelu * gelu

    total_sum = tl.sum(sum_val, axis=0)
    total_sq = tl.sum(sum_sq, axis=0)
    inv_gs = 1.0 / GROUP_SIZE
    mean = total_sum * inv_gs
    var = total_sq * inv_gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Preload per-channel weight/bias for this group
    c_offs = tl.arange(0, 16)  # CPG up to 16; we use mask
    c_mask = c_offs < CPG
    w_grp = tl.load(weight_ptr + g * CPG + c_offs, mask=c_mask, other=0.0)
    b_grp = tl.load(bias_ptr + g * CPG + c_offs, mask=c_mask, other=0.0)

    for i in tl.static_range(num_iters):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < GROUP_SIZE
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))

        c_local = offs // HW
        # gather w,b by c_local
        # Use load via pointer arithmetic
        w = tl.load(weight_ptr + g * CPG + c_local, mask=mask, other=0.0)
        b = tl.load(bias_ptr + g * CPG + c_local, mask=mask, other=0.0)

        y = (gelu - mean) * rstd * w + b
        tl.store(out_ptr + base + offs, y, mask=mask)


def fused_gelu_groupnorm(x, weight, bias, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    CPG = C // num_groups
    out = torch.empty_like(x)

    group_size = CPG * HW
    BLOCK = 2048

    grid = (N * num_groups,)
    gelu_groupnorm_kernel[grid](
        x, out, weight, bias,
        N, C, HW, num_groups, CPG,
        eps,
        GROUP_SIZE=group_size,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_gelu_groupnorm(x, self.group_norm.weight, self.group_norm.bias, self.num_groups, self.eps)
        return x