import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_mean_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    G, CPG, S,
    eps,
    inv_total,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    base = n * (G * GROUP_SIZE) + g * GROUP_SIZE

    idx = tl.arange(0, BLOCK)
    mask = idx < GROUP_SIZE

    v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
    s_sum = tl.sum(v, axis=0)
    s_sq = tl.sum(v * v, axis=0)

    mean = s_sum / GROUP_SIZE
    var = s_sq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    c_local = idx // S
    c_global = g * CPG + c_local
    w = tl.load(w_ptr + c_global, mask=mask, other=0.0)
    b = tl.load(b_ptr + c_global, mask=mask, other=0.0)

    y = (v - mean) * rstd * w + b
    y = tl.where(mask, y, 0.0)
    group_total = tl.sum(y, axis=0)
    contribution = group_total * inv_total
    tl.atomic_add(out_ptr + n, contribution)


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


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
        group_size = CPG * S
        total = C * S
        inv_total = 1.0 / total

        out = torch.zeros(N, device=x.device, dtype=x.dtype)
        BLOCK = next_pow2(group_size)

        if BLOCK >= 16384:
            num_warps = 16
        elif BLOCK >= 8192:
            num_warps = 8
        elif BLOCK >= 4096:
            num_warps = 4
        else:
            num_warps = 4

        gn_mean_kernel[(N * G,)](
            x, self.group_norm.weight, self.group_norm.bias, out,
            G, CPG, S,
            self.eps, inv_total,
            GROUP_SIZE=group_size,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=2,
        )
        return out