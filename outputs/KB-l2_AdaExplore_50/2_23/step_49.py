import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_mean_kernel(
    y_ptr, conv_b_ptr, gn_w_ptr, gn_b_ptr, out_ptr,
    OC: tl.constexpr,
    CHANS_PER_GROUP: tl.constexpr,
    SPATIAL: tl.constexpr,
    inv_group_size: tl.constexpr,
    inv_total: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)

    oc_start = g * CHANS_PER_GROUP
    base = n * OC * SPATIAL + oc_start * SPATIAL

    s_offs = tl.arange(0, BLOCK_S)

    # Preload per-channel constants
    c_offs = tl.arange(0, CHANS_PER_GROUP)
    cb_vec = tl.load(conv_b_ptr + oc_start + c_offs)
    gn_w_vec = tl.load(gn_w_ptr + oc_start + c_offs)
    gn_b_vec = tl.load(gn_b_ptr + oc_start + c_offs)

    sum_val = tl.zeros([BLOCK_S], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Phase 1: stats
    for c_idx in tl.static_range(CHANS_PER_GROUP):
        ch_base = base + c_idx * SPATIAL
        cb = tl.load(conv_b_ptr + oc_start + c_idx)
        for s_start in range(0, SPATIAL, BLOCK_S):
            s = s_start + s_offs
            s_mask = s < SPATIAL
            v = tl.load(y_ptr + ch_base + s, mask=s_mask, other=0.0) + cb
            v = tl.where(s_mask, v, 0.0)
            sum_val += v
            sum_sq += v * v

    s_total = tl.sum(sum_val, axis=0)
    ss_total = tl.sum(sum_sq, axis=0)

    mean = s_total * inv_group_size
    var = ss_total * inv_group_size - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Phase 2: normalize and accumulate output mean
    out_sum = tl.zeros([BLOCK_S], dtype=tl.float32)
    for c_idx in tl.static_range(CHANS_PER_GROUP):
        ch_base = base + c_idx * SPATIAL
        cb = tl.load(conv_b_ptr + oc_start + c_idx)
        gn_w = tl.load(gn_w_ptr + oc_start + c_idx)
        gn_b = tl.load(gn_b_ptr + oc_start + c_idx)
        scale = inv_std * gn_w
        shift = gn_b - mean * scale + cb * scale
        for s_start in range(0, SPATIAL, BLOCK_S):
            s = s_start + s_offs
            s_mask = s < SPATIAL
            v = tl.load(y_ptr + ch_base + s, mask=s_mask, other=0.0)
            normed = v * scale + shift
            normed_masked = tl.where(s_mask, normed, 0.0)
            out_sum += normed_masked

    result = tl.sum(out_sum, axis=0) * inv_total
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

        y = F.conv3d(x, self.conv.weight, None)
        y = y.contiguous().view(N, OC, SPATIAL)

        conv_b = self.conv.bias.contiguous()
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        BLOCK_S = 1024

        grid = (N, self.num_groups)
        gn_mean_kernel[grid](
            y, conv_b, gn_w, gn_b, out,
            OC,
            CHANS_PER_GROUP,
            SPATIAL,
            1.0 / GROUP_SIZE,
            1.0 / (OC * SPATIAL),
            eps,
            BLOCK_S,
            num_warps=8,
            num_stages=3,
        )

        return out