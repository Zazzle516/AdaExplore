import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mean_reduce_kernel(
    x_ptr, out_ptr,
    N, C, S,
    inv_total,
    BLOCK_S: tl.constexpr,
):
    # one program per batch
    n = tl.program_id(0)
    total_per_n = C * S
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    base = n * total_per_n
    # loop over C*S in chunks
    for off in range(0, total_per_n, BLOCK_S):
        idx = off + tl.arange(0, BLOCK_S)
        mask = idx < total_per_n
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        acc += v
    s = tl.sum(acc, axis=0)
    tl.store(out_ptr + n, s * inv_total)


@triton.jit
def group_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, G, CPG, S, eps,
    BLOCK: tl.constexpr,
):
    # one program per (n, g)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    group_size = CPG * S
    base = n * (G * CPG * S) + g * CPG * S

    # compute mean and var
    sum_v = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_v += v
        sum_sq += v * v
    mean = tl.sum(sum_v, axis=0) / group_size
    var = tl.sum(sum_sq, axis=0) / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # apply normalization with per-channel weight/bias
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        # channel index within group
        c_local = idx // S
        c_global = g * CPG + c_local
        w = tl.load(w_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(b_ptr + c_global, mask=mask, other=0.0)
        y = (v - mean) * rstd * w + b
        tl.store(out_ptr + base + idx, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        N, C, D, H, W = x.shape
        S = D * H * W
        x = x.contiguous()

        # group norm
        y = torch.empty_like(x)
        G = self.num_groups
        CPG = C // G
        BLOCK = 1024
        grid = (N * G,)
        group_norm_kernel[grid](
            x, self.group_norm.weight, self.group_norm.bias, y,
            N, G, CPG, S, self.eps,
            BLOCK=BLOCK, num_warps=4,
        )

        # mean reduction across C, D, H, W
        out = torch.empty(N, device=x.device, dtype=x.dtype)
        total = C * S
        inv_total = 1.0 / total
        BLOCK_S = 1024
        mean_reduce_kernel[(N,)](
            y, out, N, C, S, inv_total,
            BLOCK_S=BLOCK_S, num_warps=4,
        )
        return out