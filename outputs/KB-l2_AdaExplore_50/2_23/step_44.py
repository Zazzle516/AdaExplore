import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mean_reduce_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    # one program per batch
    n = tl.program_id(0)
    total = C * S
    acc = 0.0
    # iterate over all elements in this batch
    base = n * total
    num_blocks = (total + BLOCK_S - 1) // BLOCK_S
    for b in range(0, num_blocks):
        offs = b * BLOCK_S + tl.arange(0, BLOCK_S)
        mask = offs < total
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(v, axis=0)
    tl.store(out_ptr + n, acc / total)


@triton.jit
def group_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, G, CPG, S,
    eps,
    BLOCK_S: tl.constexpr,
):
    # one program per (n, g) - computes mean/var over CPG*S elements and writes normalized output
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_size = CPG * S
    base = n * (G * CPG * S) + g * (CPG * S)

    # compute mean and var with two pass
    sum_val = 0.0
    sum_sq = 0.0
    num_blocks = (group_size + BLOCK_S - 1) // BLOCK_S
    for b in range(0, num_blocks):
        offs = b * BLOCK_S + tl.arange(0, BLOCK_S)
        mask = offs < group_size
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        sum_val += tl.sum(v, axis=0)
        sum_sq += tl.sum(v * v, axis=0)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # apply normalization with weight and bias
    for b in range(0, num_blocks):
        offs = b * BLOCK_S + tl.arange(0, BLOCK_S)
        mask = offs < group_size
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        # channel index within group
        c_in_group = offs // S
        c = g * CPG + c_in_group
        w = tl.load(weight_ptr + c, mask=mask, other=0.0)
        bi = tl.load(bias_ptr + c, mask=mask, other=0.0)
        out = (v - mean) * rstd * w + bi
        tl.store(out_ptr + base + offs, out, mask=mask)


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

        out_gn = torch.empty_like(x)

        # choose BLOCK_S
        group_size = CPG * S
        BLOCK_S = 1024
        if group_size < 256:
            BLOCK_S = 128
        elif group_size < 1024:
            BLOCK_S = 256

        weight = self.group_norm.weight.contiguous()
        bias = self.group_norm.bias.contiguous()

        grid_gn = (N * G,)
        group_norm_kernel[grid_gn](
            x, weight, bias, out_gn,
            N, G, CPG, S,
            self.eps,
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )

        # mean over C, D, H, W
        out = torch.empty(N, device=x.device, dtype=x.dtype)
        BLOCK_M = 1024
        grid_m = (N,)
        mean_reduce_kernel[grid_m](
            out_gn, out,
            N, C, S,
            BLOCK_S=BLOCK_M,
            num_warps=4,
        )
        return out