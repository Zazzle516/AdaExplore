import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_S': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=3),
    ],
    key=['C_PER_GROUP', 'S'],
)
@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    group_size = C_PER_GROUP * S
    base = n * C * S + g * C_PER_GROUP * S

    offs = tl.arange(0, BLOCK_S)

    sum_x = tl.zeros([BLOCK_S], dtype=tl.float32)
    sum_x2 = tl.zeros([BLOCK_S], dtype=tl.float32)

    NUM_BLOCKS = (S + BLOCK_S - 1) // BLOCK_S

    # Pass 1: compute sum/sum2, store ReLU'd values to out buffer (scratch)
    for c in range(0, C_PER_GROUP):
        c_offset = base + c * S
        for b in range(0, NUM_BLOCKS):
            s_start = b * BLOCK_S
            cur = s_start + offs
            mask = cur < S
            v = tl.load(x_ptr + c_offset + cur, mask=mask, other=0.0)
            v = tl.maximum(v, 0.0)
            sum_x += v
            sum_x2 += v * v
            tl.store(out_ptr + c_offset + cur, v, mask=mask)

    total_sum = tl.sum(sum_x, axis=0)
    total_sum2 = tl.sum(sum_x2, axis=0)

    inv_gs = 1.0 / group_size
    mean = total_sum * inv_gs
    var = total_sum2 * inv_gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: read ReLU'd values from out, normalize, write back
    for c in range(0, C_PER_GROUP):
        c_idx = g * C_PER_GROUP + c
        w = tl.load(weight_ptr + c_idx)
        bb = tl.load(bias_ptr + c_idx)
        c_offset = base + c * S
        scale = rstd * w
        shift = bb - mean * scale
        for b in range(0, NUM_BLOCKS):
            s_start = b * BLOCK_S
            cur = s_start + offs
            mask = cur < S
            v = tl.load(out_ptr + c_offset + cur, mask=mask, other=0.0)
            y = v * scale + shift
            tl.store(out_ptr + c_offset + cur, y, mask=mask)


def fused_relu_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    C_PER_GROUP = C // groups
    x_flat = x.contiguous().view(N, C, S)
    out = torch.empty_like(x_flat)

    grid = (N * groups,)
    fused_relu_groupnorm_kernel[grid](
        x_flat, out, weight, bias,
        N, C, S,
        GROUPS=groups,
        C_PER_GROUP=C_PER_GROUP,
        eps=eps,
    )
    return out.view(N, C, D, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
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