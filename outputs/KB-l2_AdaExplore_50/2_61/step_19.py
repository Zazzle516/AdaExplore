import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    C, S, CPG,
    eps,
    BLOCK_S: tl.constexpr,
    CPG_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size = CPG * S
    base = pid_n * C * S + pid_g * CPG * S

    sum_val = 0.0
    sum_sq = 0.0

    for c_off in tl.static_range(0, CPG_CONST):
        row_base = base + c_off * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            sum_val += tl.sum(x, axis=0)
            sum_sq += tl.sum(x * x, axis=0)

    inv_n = 1.0 / group_size
    mean = sum_val * inv_n
    var = sum_sq * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for c_off in tl.static_range(0, CPG_CONST):
        c_idx = pid_g * CPG + c_off
        w = tl.load(weight_ptr + c_idx)
        b = tl.load(bias_ptr + c_idx)
        row_base = base + c_off * S
        scale = rstd * w
        shift = b - mean * scale
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            y = x * scale + shift
            tl.store(y_ptr + row_base + offs, y, mask=mask)


@triton.jit
def fused_relu_groupnorm_kernel_generic(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    C, S, CPG,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size = CPG * S
    base = pid_n * C * S + pid_g * CPG * S

    sum_val = 0.0
    sum_sq = 0.0

    for c_off in range(0, CPG):
        row_base = base + c_off * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            sum_val += tl.sum(x, axis=0)
            sum_sq += tl.sum(x * x, axis=0)

    inv_n = 1.0 / group_size
    mean = sum_val * inv_n
    var = sum_sq * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for c_off in range(0, CPG):
        c_idx = pid_g * CPG + c_off
        w = tl.load(weight_ptr + c_idx)
        b = tl.load(bias_ptr + c_idx)
        row_base = base + c_off * S
        scale = rstd * w
        shift = b - mean * scale
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            y = x * scale + shift
            tl.store(y_ptr + row_base + offs, y, mask=mask)


def fused_relu_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    CPG = C // groups
    x_c = x.contiguous()
    y = torch.empty_like(x_c)

    grid = (N, groups)
    # S = 34*34*34 = 39304. BLOCK_S=8192 -> 5 iterations
    if CPG == 16 and S <= 65536:
        BLOCK_S = 8192
        fused_relu_groupnorm_kernel[grid](
            x_c, y, weight, bias,
            C, S, CPG,
            eps,
            BLOCK_S=BLOCK_S,
            CPG_CONST=16,
            num_warps=8,
            num_stages=3,
        )
    else:
        BLOCK_S = 4096
        fused_relu_groupnorm_kernel_generic[grid](
            x_c, y, weight, bias,
            C, S, CPG,
            eps,
            BLOCK_S=BLOCK_S,
            num_warps=8,
            num_stages=3,
        )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.groups = groups
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_relu_groupnorm(x, self.group_norm.weight, self.group_norm.bias, self.groups, self.eps)
        return x