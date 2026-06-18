import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, y_ptr, gamma_ptr, beta_ptr,
    S,
    C_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    GROUPS = tl.num_programs(1)
    C = GROUPS * C_PER_GROUP

    base = n * C * S + g * C_PER_GROUP * S

    sum_val = tl.zeros([], dtype=tl.float32)
    sum_sq = tl.zeros([], dtype=tl.float32)

    offs_s = tl.arange(0, BLOCK_S)
    for c in tl.static_range(0, C_PER_GROUP):
        c_off = base + c * S
        for s_start in range(0, S, BLOCK_S):
            s_idx = s_start + offs_s
            mask = s_idx < S
            x = tl.load(x_ptr + c_off + s_idx, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            sum_val += tl.sum(x)
            sum_sq += tl.sum(x * x)

    inv_n = 1.0 / GROUP_SIZE
    mean = sum_val * inv_n
    var = sum_sq * inv_n - mean * mean
    rstd = tl.rsqrt(var + eps)

    for c in tl.static_range(0, C_PER_GROUP):
        c_global = g * C_PER_GROUP + c
        gamma = tl.load(gamma_ptr + c_global)
        beta = tl.load(beta_ptr + c_global)
        scale = rstd * gamma
        bias_term = beta - mean * scale
        c_off = base + c * S
        for s_start in range(0, S, BLOCK_S):
            s_idx = s_start + offs_s
            mask = s_idx < S
            x = tl.load(x_ptr + c_off + s_idx, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            y = x * scale + bias_term
            tl.store(y_ptr + c_off + s_idx, y, mask=mask)


def fused_relu_groupnorm(x, gamma, beta, groups, eps=1e-5):
    assert x.is_cuda and x.is_contiguous()
    N, C = x.shape[0], x.shape[1]
    spatial = x.shape[2:]
    S = 1
    for d in spatial:
        S *= d
    C_per_group = C // groups
    group_size = C_per_group * S
    y = torch.empty_like(x)

    BLOCK_S = 1024
    grid = (N, groups)
    fused_relu_groupnorm_kernel[grid](
        x, y, gamma, beta,
        S,
        C_PER_GROUP=C_per_group,
        BLOCK_S=BLOCK_S,
        eps=eps,
        GROUP_SIZE=group_size,
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
        self.out_channels = out_channels
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        y = fused_relu_groupnorm(x, gamma, beta, self.groups, self.eps)
        return y