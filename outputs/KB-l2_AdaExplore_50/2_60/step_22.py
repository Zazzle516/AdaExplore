import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
    ],
    key=['S', 'C_PER_G_CONST'],
)
@triton.jit
def swish_groupnorm_hardswish_kernel(
    x_ptr, y_ptr,
    gamma_ptr, beta_ptr,
    N, C, S,
    G,
    eps,
    inv_group_size,
    BLOCK_S: tl.constexpr,
    C_PER_G_CONST: tl.constexpr,
):
    # one program per (n, g)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    base = n * C * S + g * C_PER_G_CONST * S

    # Pass 1: compute mean and var of swish(x) over group.
    # Store swish(x) into y as scratch to avoid recomputing sigmoid in pass 2.
    sum_val = tl.zeros([], dtype=tl.float32)
    sum_sq = tl.zeros([], dtype=tl.float32)

    for c_inner in tl.static_range(0, C_PER_G_CONST):
        c_off = base + c_inner * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + c_off + offs, mask=mask, other=0.0).to(tl.float32)
            sw = x * tl.sigmoid(x)
            sw_m = tl.where(mask, sw, 0.0)
            sum_val += tl.sum(sw_m, axis=0)
            sum_sq += tl.sum(sw_m * sw_m, axis=0)
            tl.store(y_ptr + c_off + offs, sw, mask=mask)

    mean = sum_val * inv_group_size
    var = sum_sq * inv_group_size - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Pass 2: load swish from y, normalize, affine, hardswish, store
    for c_inner in tl.static_range(0, C_PER_G_CONST):
        c_idx = g * C_PER_G_CONST + c_inner
        gamma = tl.load(gamma_ptr + c_idx).to(tl.float32)
        beta = tl.load(beta_ptr + c_idx).to(tl.float32)
        c_off = base + c_inner * S
        a = gamma * rstd
        b = beta - mean * a
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            sw = tl.load(y_ptr + c_off + offs, mask=mask, other=0.0).to(tl.float32)
            y = sw * a + b
            t = y + 3.0
            t = tl.minimum(tl.maximum(t, 0.0), 6.0)
            out = y * t * (1.0 / 6.0)
            tl.store(y_ptr + c_off + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.groups = groups
        self.out_channels = out_channels
        self.eps = eps

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape
        S = D * H * W
        x_c = x.contiguous()
        y = torch.empty_like(x_c)

        G = self.groups
        C_PER_G = C // G

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        inv_group_size = 1.0 / float(C_PER_G * S)

        grid = (N * G,)
        swish_groupnorm_hardswish_kernel[grid](
            x_c, y,
            gamma, beta,
            N, C, S,
            G,
            self.eps,
            inv_group_size,
            C_PER_G_CONST=C_PER_G,
        )
        return y