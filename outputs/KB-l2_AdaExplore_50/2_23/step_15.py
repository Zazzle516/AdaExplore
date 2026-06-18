import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_mean_kernel(
    y_ptr, gn_w_ptr, gn_b_ptr, out_ptr,
    OC: tl.constexpr,
    CHANS_PER_GROUP: tl.constexpr,
    SPATIAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)

    oc_start = g * CHANS_PER_GROUP
    base = n * OC * SPATIAL + oc_start * SPATIAL

    s_offs = tl.arange(0, BLOCK_S)

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for c_idx in tl.static_range(CHANS_PER_GROUP):
        ch_base = base + c_idx * SPATIAL
        for s_start in range(0, SPATIAL, BLOCK_S):
            s = s_start + s_offs
            s_mask = s < SPATIAL
            v = tl.load(y_ptr + ch_base + s, mask=s_mask, other=0.0)
            sum_val += tl.sum(v, axis=0)
            sum_sq += tl.sum(v * v, axis=0)

    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Fused: out_sum = sum_c [ scale_c * sum_c_v + shift_c * SPATIAL ]
    # where scale_c = inv_std * gn_w_c, shift_c = gn_b_c - mean * scale_c
    # We already have per-channel sums implicitly. Recompute per-channel sums in pass 2-lite:
    # Actually we can do it without re-reading y by tracking per-channel sum in pass 1.
    # But that requires CHANS_PER_GROUP separate accumulators. Use static_range with arrays.
    # Simpler: do a second pass reading y again. But we want to avoid that.
    # Use the identity: sum_c (scale_c * S_c + shift_c * SPATIAL) where S_c = sum over spatial of v_c
    # We need S_c. Let's just compute it in a separate loop accumulating per-channel.

    # Reload to compute per-channel sums and final.
    out_sum = tl.zeros((), dtype=tl.float32)
    for c_idx in tl.static_range(CHANS_PER_GROUP):
        ch_base = base + c_idx * SPATIAL
        gn_w = tl.load(gn_w_ptr + oc_start + c_idx)
        gn_b = tl.load(gn_b_ptr + oc_start + c_idx)
        scale = inv_std * gn_w
        shift = gn_b - mean * scale
        s_c = tl.zeros((), dtype=tl.float32)
        for s_start in range(0, SPATIAL, BLOCK_S):
            s = s_start + s_offs
            s_mask = s < SPATIAL
            v = tl.load(y_ptr + ch_base + s, mask=s_mask, other=0.0)
            s_c += tl.sum(v, axis=0)
        out_sum += scale * s_c + shift * SPATIAL

    total = OC * SPATIAL
    result = out_sum / total
    tl.atomic_add(out_ptr + n, result)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        N, IC, D_in, H_in, W_in = x.shape
        K = self.kernel_size
        OC = self.out_channels
        D_out = D_in - K + 1
        H_out = H_in - K + 1
        W_out = W_in - K + 1
        SPATIAL = D_out * H_out * W_out
        CHANS_PER_GROUP = OC // self.num_groups
        GROUP_SIZE = CHANS_PER_GROUP * SPATIAL

        y = F.conv3d(x, self.conv.weight, self.conv.bias)
        y = y.contiguous().view(N, OC, SPATIAL)

        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        BLOCK_S = 2048

        grid = (N, self.num_groups)
        gn_mean_kernel[grid](
            y, gn_w, gn_b, out,
            OC,
            CHANS_PER_GROUP,
            SPATIAL,
            GROUP_SIZE,
            eps,
            BLOCK_S,
            num_warps=8,
            num_stages=3,
        )

        return out