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

    # Single pass: compute group sum, sum_sq, AND per-channel sums.
    # We use a 1D buffer indexed via static_range channels by accumulating
    # into a [CHANS_PER_GROUP] vector via tl.arange masking trick is hard;
    # instead, store per-channel partial sums inline using static_range.
    # We'll compute per-channel sum_c during pass 1 and use them in epilogue.
    
    # Use a tensor of shape [CHANS_PER_GROUP] for per-channel sums
    per_ch_sum = tl.zeros((CHANS_PER_GROUP,), dtype=tl.float32)

    for s_start in range(0, SPATIAL, BLOCK_S):
        s = s_start + s_offs
        s_mask = s < SPATIAL
        # Accumulate a 2D tile: [CHANS_PER_GROUP, BLOCK_S]
        for c_idx in tl.static_range(CHANS_PER_GROUP):
            v = tl.load(y_ptr + base + c_idx * SPATIAL + s, mask=s_mask, other=0.0)
            sv = tl.sum(v, axis=0)
            sum_val += sv
            sum_sq += tl.sum(v * v, axis=0)
            # accumulate into per_ch_sum[c_idx]
            mask_vec = tl.arange(0, CHANS_PER_GROUP) == c_idx
            per_ch_sum += tl.where(mask_vec, sv, 0.0)

    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Load gn weights/bias for this group
    c_offs = tl.arange(0, CHANS_PER_GROUP)
    gn_w = tl.load(gn_w_ptr + oc_start + c_offs)
    gn_b = tl.load(gn_b_ptr + oc_start + c_offs)
    scale = inv_std * gn_w
    shift = gn_b - mean * scale

    # out_sum = sum_c (scale_c * sum_c + shift_c * SPATIAL)
    out_sum = tl.sum(scale * per_ch_sum + shift * SPATIAL, axis=0)

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