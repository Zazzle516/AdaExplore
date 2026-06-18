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
    key=['group_size'],
)
@triton.jit
def group_norm_mean_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, G, CPG, S,
    group_size,
    inv_total,
    eps,
    BLOCK_S: tl.constexpr,
):
    # one program per (n, g)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    base = n * (G * CPG * S) + g * (CPG * S)

    # compute mean and var
    sum_val = 0.0
    sum_sq = 0.0
    num_blocks = (group_size + BLOCK_S - 1) // BLOCK_S
    for b in range(0, num_blocks):
        offs = b * BLOCK_S + tl.arange(0, BLOCK_S)
        mask = offs < group_size
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        sum_val += tl.sum(v, axis=0)
        sum_sq += tl.sum(v * v, axis=0)

    inv_gs = 1.0 / group_size
    mean = sum_val * inv_gs
    var = sum_sq * inv_gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # second pass: accumulate sum of normalized*weight + bias
    acc = 0.0
    for b in range(0, num_blocks):
        offs = b * BLOCK_S + tl.arange(0, BLOCK_S)
        mask = offs < group_size
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        c_in_group = offs // S
        c = g * CPG + c_in_group
        w = tl.load(weight_ptr + c, mask=mask, other=0.0)
        bi = tl.load(bias_ptr + c, mask=mask, other=0.0)
        scale = w * rstd
        shift = bi - mean * scale
        out = v * scale + shift
        out = tl.where(mask, out, 0.0)
        acc += tl.sum(out, axis=0)

    # atomic add scaled by 1/(C*S) -> use inv_total
    tl.atomic_add(out_ptr + n, acc * inv_total)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.eps = 1e-5

    def forward(self, x):
        x = x.cuda()
        # conv
        x = self.conv(x)
        x = x.contiguous()

        N, C, D, H, W = x.shape
        S = D * H * W
        G = self.num_groups
        CPG = C // G
        group_size = CPG * S
        total = C * S
        inv_total = 1.0 / total

        weight = self.group_norm.weight.contiguous()
        bias = self.group_norm.bias.contiguous()

        # zero-init output for atomic accumulation
        out = torch.zeros(N, device=x.device, dtype=x.dtype)

        grid_gn = (N * G,)
        group_norm_mean_kernel[grid_gn](
            x, weight, bias, out,
            N, G, CPG, S,
            group_size,
            inv_total,
            self.eps,
        )
        return out