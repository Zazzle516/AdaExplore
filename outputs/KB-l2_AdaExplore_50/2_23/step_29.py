import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 8192}, num_warps=16, num_stages=2),
    ],
    key=['S', 'CPG'],
)
@triton.jit
def gn_mean_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    G, CPG: tl.constexpr, S,
    eps, inv_total,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    base = n * (G * CPG * S) + g * CPG * S

    # Per-channel sum and sumsq accumulators (CPG is small constexpr, e.g., 3)
    # Single pass over S, accumulate per-channel.
    # Use 2D layout: [CPG, BLOCK_S]
    sum_c = tl.zeros([CPG], dtype=tl.float32)
    sumsq_c = tl.zeros([CPG], dtype=tl.float32)

    offs_c = tl.arange(0, CPG)
    for off in range(0, S, BLOCK_S):
        offs_s = off + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S
        # ptr: base + c * S + s
        ptrs = base + offs_c[:, None] * S + offs_s[None, :]
        v = tl.load(x_ptr + ptrs, mask=mask_s[None, :], other=0.0)
        sum_c += tl.sum(v, axis=1)
        sumsq_c += tl.sum(v * v, axis=1)

    group_size = CPG * S
    total_sum = tl.sum(sum_c, axis=0)
    total_sq = tl.sum(sumsq_c, axis=0)
    mean = total_sum / group_size
    var = total_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # sum over group of y_{c,s} = (x_{c,s} - mean) * rstd * w_c + b_c
    # = rstd * sum_c w_c * (S_c - mean * S) + S * sum_c b_c
    c_global = g * CPG + offs_c
    w = tl.load(w_ptr + c_global)
    b = tl.load(b_ptr + c_global)
    per_c = rstd * w * (sum_c - mean * S) + b * S
    group_total = tl.sum(per_c, axis=0)
    contribution = group_total * inv_total
    tl.atomic_add(out_ptr + n, contribution)


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
        x = x.contiguous()
        N, C, D, H, W = x.shape
        S = D * H * W
        G = self.num_groups
        CPG = C // G
        total = C * S
        inv_total = 1.0 / total

        out = torch.zeros(N, device=x.device, dtype=x.dtype)
        gn_mean_fused_kernel[(N * G,)](
            x, self.group_norm.weight, self.group_norm.bias, out,
            G, CPG, S,
            self.eps, inv_total,
        )
        return out