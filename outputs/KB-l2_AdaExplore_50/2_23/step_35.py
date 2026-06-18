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

    # Single pass: compute sum and sum_sq
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

    # Use closed form: mean over group of (v - mean)*rstd*w + b
    # = rstd * (sum(v*w) - mean*sum(w))/gs + sum(b)/gs ... but we need overall
    # sum across full group then add to atomic.
    # sum_y = sum((v-mean)*rstd*w_c + b_c) over c in group, s in S
    # = rstd * sum_c w_c * (sum_s v[c,s] - mean*S) + S*sum_c b_c
    # We need per-channel sums of v. Compute them via second pass with c info,
    # OR compute weight-weighted sum during stat pass.
    # Simpler: do a second pass accumulating per-element y.
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        c_local = idx // S
        c_global = g * CPG + c_local
        w = tl.load(w_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(b_ptr + c_global, mask=mask, other=0.0)
        y = (v - mean) * rstd * w + b
        y = tl.where(mask, y, 0.0)
        acc += y
    group_sum = tl.sum(acc, axis=0)
    tl.atomic_add(out_ptr + n, group_sum * inv_total)


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
        group_size = CPG * S
        # Pick BLOCK to balance occupancy. group_size = 3 * 22*30*30 = 59400
        BLOCK = 8192
        grid = (N * G,)
        group_norm_mean_kernel[grid](
            x, self.group_norm.weight, self.group_norm.bias, out,
            G, CPG, S,
            self.eps, inv_total,
            BLOCK=BLOCK, num_warps=8, num_stages=2,
        )
        return out