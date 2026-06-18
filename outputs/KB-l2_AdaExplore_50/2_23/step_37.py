import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
    ],
    key=['SPATIAL', 'CHANS_PER_GROUP'],
)
@triton.jit
def gn_mean_kernel(
    y_ptr, conv_b_ptr, gn_w_ptr,
    sum_cb_g_ptr, sum_cb_sq_g_ptr, sum_gn_w_g_ptr,
    sum_gn_b_g_ptr, sum_gn_w_cb_g_ptr,
    partial_ptr,
    OC: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
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

    total_S = 0.0
    total_S2 = 0.0
    sum_cb_v = 0.0
    weighted_S = 0.0

    for c_idx in tl.static_range(CHANS_PER_GROUP):
        ch_base = base + c_idx * SPATIAL
        cb = tl.load(conv_b_ptr + oc_start + c_idx)
        gw = tl.load(gn_w_ptr + oc_start + c_idx)
        for s_start in range(0, SPATIAL, BLOCK_S):
            s = s_start + s_offs
            s_mask = s < SPATIAL
            v = tl.load(y_ptr + ch_base + s, mask=s_mask, other=0.0)
            v = tl.where(s_mask, v, 0.0)
            sv = tl.sum(v, axis=0)
            sv2 = tl.sum(v * v, axis=0)
            total_S += sv
            total_S2 += sv2
            sum_cb_v += cb * sv
            weighted_S += gw * sv

    sum_cb = tl.load(sum_cb_g_ptr + g)
    sum_cb_sq = tl.load(sum_cb_sq_g_ptr + g)
    sum_gn_w = tl.load(sum_gn_w_g_ptr + g)
    sum_gn_b = tl.load(sum_gn_b_g_ptr + g)
    sum_gn_w_cb = tl.load(sum_gn_w_cb_g_ptr + g)

    SPATIAL_F = SPATIAL + 0.0
    total_sum_b = total_S + sum_cb * SPATIAL_F
    mean = total_sum_b / GROUP_SIZE
    total_sumsq_b = total_S2 + 2.0 * sum_cb_v + SPATIAL_F * sum_cb_sq
    var = total_sumsq_b / GROUP_SIZE - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    out_sum = inv_std * weighted_S \
              + inv_std * SPATIAL_F * sum_gn_w_cb \
              - inv_std * SPATIAL_F * mean * sum_gn_w \
              + SPATIAL_F * sum_gn_b

    tl.store(partial_ptr + n * NUM_GROUPS + g, out_sum)


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
        inv_total = 1.0 / float(OC * SPATIAL)

        # Per-group precomputed scalars (size NUM_GROUPS each)
        cb_g = conv_b.view(self.num_groups, CHANS_PER_GROUP)
        gw_g = gn_w.view(self.num_groups, CHANS_PER_GROUP)
        gb_g = gn_b.view(self.num_groups, CHANS_PER_GROUP)
        sum_cb = cb_g.sum(dim=1).contiguous()
        sum_cb_sq = (cb_g * cb_g).sum(dim=1).contiguous()
        sum_gn_w = gw_g.sum(dim=1).contiguous()
        sum_gn_b = gb_g.sum(dim=1).contiguous()
        sum_gn_w_cb = (gw_g * cb_g).sum(dim=1).contiguous()

        partial = torch.empty((N, self.num_groups), device=x.device, dtype=torch.float32)

        grid = (N, self.num_groups)
        gn_mean_kernel[grid](
            y, conv_b, gn_w,
            sum_cb, sum_cb_sq, sum_gn_w, sum_gn_b, sum_gn_w_cb,
            partial,
            OC,
            self.num_groups,
            CHANS_PER_GROUP,
            SPATIAL,
            GROUP_SIZE,
            eps,
        )

        out = partial.sum(dim=1) * inv_total
        return out