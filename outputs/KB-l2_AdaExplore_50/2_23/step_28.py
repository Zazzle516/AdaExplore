import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_mean_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    G, CPG, S,
    eps, inv_total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    group_size = CPG * S
    base = n * (G * CPG * S) + g * CPG * S

    # Single-pass Welford-like: accumulate sum and sumsq in registers
    sum_v = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)

    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_v += v
        sum_sq += v * v

    s_sum = tl.sum(sum_v, axis=0)
    s_sq = tl.sum(sum_sq, axis=0)
    mean = s_sum / group_size
    var = s_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: compute (x-mean)*rstd*w + b summed over the group.
    # sum_{c,s} ((x - mean) * rstd * w_c + b_c)
    #   = rstd * sum_c w_c * (sum_s x_{c,s}) - rstd * mean * sum_c w_c * S + sum_c b_c * S
    # But easier: just reload and accumulate.
    partial = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        c_local = idx // S
        c_global = g * CPG + c_local
        w = tl.load(w_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(b_ptr + c_global, mask=mask, other=0.0)
        y = (v - mean) * rstd * w + b
        partial += tl.where(mask, y, 0.0)

    group_total = tl.sum(partial, axis=0)
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
        BLOCK = 2048
        gn_mean_fused_kernel[(N * G,)](
            x, self.group_norm.weight, self.group_norm.bias, out,
            G, CPG, S,
            self.eps, inv_total,
            BLOCK=BLOCK, num_warps=8, num_stages=3,
        )
        return out