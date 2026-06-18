import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
    ],
    key=['C_PER_GROUP', 'S'],
)
@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, y_ptr, gamma_ptr, beta_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)

    group_size = C_PER_GROUP * S
    base = n * C * S + g * C_PER_GROUP * S

    sum_val = tl.zeros([], dtype=tl.float32)
    sum_sq = tl.zeros([], dtype=tl.float32)

    offs_s = tl.arange(0, BLOCK_S)
    for c in range(0, C_PER_GROUP):
        c_off = base + c * S
        for s_start in range(0, S, BLOCK_S):
            s_idx = s_start + offs_s
            mask = s_idx < S
            x = tl.load(x_ptr + c_off + s_idx, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            tl.store(y_ptr + c_off + s_idx, x, mask=mask)
            sum_val += tl.sum(tl.where(mask, x, 0.0))
            sum_sq += tl.sum(tl.where(mask, x * x, 0.0))

    inv_n = 1.0 / group_size
    mean = sum_val * inv_n
    var = sum_sq * inv_n - mean * mean
    rstd = tl.rsqrt(var + eps)

    for c in range(0, C_PER_GROUP):
        c_global = g * C_PER_GROUP + c
        gamma = tl.load(gamma_ptr + c_global)
        beta = tl.load(beta_ptr + c_global)
        c_off = base + c * S
        a = gamma * rstd
        b = beta - mean * a
        for s_start in range(0, S, BLOCK_S):
            s_idx = s_start + offs_s
            mask = s_idx < S
            x = tl.load(y_ptr + c_off + s_idx, mask=mask, other=0.0)
            y = x * a + b
            tl.store(y_ptr + c_off + s_idx, y, mask=mask)


def fused_relu_groupnorm(x, gamma, beta, groups, eps=1e-5):
    assert x.is_cuda and x.is_contiguous()
    N, C = x.shape[0], x.shape[1]
    spatial = x.shape[2:]
    S = 1
    for d in spatial:
        S *= d
    C_per_group = C // groups
    y = torch.empty_like(x)

    grid = (N, groups)
    fused_relu_groupnorm_kernel[grid](
        x, y, gamma, beta,
        N, C, S,
        GROUPS=groups,
        C_PER_GROUP=C_per_group,
        eps=eps,
    )
    return y.view(N, C, *spatial)


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