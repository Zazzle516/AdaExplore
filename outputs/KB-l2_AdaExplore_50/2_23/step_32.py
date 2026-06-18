import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def group_norm_mean_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    G, CPG, S,
    eps: tl.constexpr,
    inv_total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    group_size = CPG * S
    base = n * (G * CPG * S) + g * CPG * S

    sum_v = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_v += v
        sum_sq += v * v
    s_v = tl.sum(sum_v, axis=0)
    s_sq = tl.sum(sum_sq, axis=0)
    inv_gs = 1.0 / group_size
    mean = s_v * inv_gs
    var = s_sq * inv_gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # sum_w_per_c = sum_{s} 1 = S, so for each channel c in group:
    # sum over s of ((v - mean)*rstd*w_c + b_c) = (sum_v_c - S*mean)*rstd*w_c + S*b_c
    # We need per-channel sums. Compute by iterating channels.
    acc_total = 0.0
    # accumulate per-channel
    for c_local in range(0, CPG):
        c_global = g * CPG + c_local
        w = tl.load(w_ptr + c_global)
        b = tl.load(b_ptr + c_global)
        # sum v over s for this channel
        c_base = base + c_local * S
        sumc = tl.zeros([BLOCK], dtype=tl.float32)
        for off in range(0, S, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            mask = idx < S
            v = tl.load(x_ptr + c_base + idx, mask=mask, other=0.0)
            sumc += v
        sv_c = tl.sum(sumc, axis=0)
        contrib = (sv_c - S * mean) * rstd * w + S * b
        acc_total += contrib

    tl.atomic_add(out_ptr + n, acc_total * inv_total)


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

        G = self.num_groups
        CPG = C // G
        total = C * S
        inv_total = 1.0 / total

        out = torch.zeros(N, device=x.device, dtype=x.dtype)
        BLOCK = 2048
        grid = (N * G,)
        group_norm_mean_kernel[grid](
            x, self.group_norm.weight, self.group_norm.bias, out,
            G, CPG, S,
            self.eps, inv_total,
            BLOCK=BLOCK, num_warps=8, num_stages=3,
        )
        return out