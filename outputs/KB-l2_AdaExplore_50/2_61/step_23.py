import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # Each program handles one (batch, group)
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    group_size = C_PER_GROUP * S
    base = n * C * S + g * C_PER_GROUP * S

    # 2D tile over (C_PER_GROUP, BLOCK_S)
    c_offs = tl.arange(0, C_PER_GROUP)  # [C_PER_GROUP]
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    # Pass 1: ReLU + reduction, cache to out buffer
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)  # [BLOCK_S]
        mask_s = s_offs < S
        ptrs = x_ptr + base + c_offs[:, None] * S + s_offs[None, :]
        mask = mask_s[None, :]
        v = tl.load(ptrs, mask=mask, other=0.0)
        v = tl.maximum(v, 0.0)
        # store post-ReLU to scratch (out buffer)
        out_ptrs = out_ptr + base + c_offs[:, None] * S + s_offs[None, :]
        tl.store(out_ptrs, v, mask=mask)
        sum_x += tl.sum(v, axis=0).to(tl.float32).sum(axis=0) if False else tl.sum(v)
        sum_x2 += tl.sum(v * v)

    mean = sum_x / group_size
    var = sum_x2 / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Load weight/bias for all channels in group
    w_idx = g * C_PER_GROUP + c_offs
    w = tl.load(weight_ptr + w_idx)  # [C_PER_GROUP]
    b = tl.load(bias_ptr + w_idx)    # [C_PER_GROUP]

    # Pass 2: normalize from cached post-ReLU buffer
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask_s = s_offs < S
        out_ptrs = out_ptr + base + c_offs[:, None] * S + s_offs[None, :]
        mask = mask_s[None, :]
        v = tl.load(out_ptrs, mask=mask, other=0.0)
        y = (v - mean) * rstd * w[:, None] + b[:, None]
        tl.store(out_ptrs, y, mask=mask)


def fused_relu_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    C_PER_GROUP = C // groups
    x_flat = x.contiguous().view(N, C, S)
    out = torch.empty_like(x_flat)

    BLOCK_S = 512
    grid = (N * groups,)
    fused_relu_groupnorm_kernel[grid](
        x_flat, out, weight, bias,
        N, C, S,
        GROUPS=groups,
        C_PER_GROUP=C_PER_GROUP,
        eps=eps,
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=2,
    )
    return out.view(N, C, D, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.groups = groups
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_relu_groupnorm(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.groups,
            self.eps,
        )
        return x