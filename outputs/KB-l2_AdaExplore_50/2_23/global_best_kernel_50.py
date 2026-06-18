import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_S': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_S': 8192}, num_warps=16, num_stages=3),
    ],
    key=['SPATIAL', 'CHANS_PER_GROUP'],
)
@triton.jit
def gn_mean_kernel(
    y_ptr, conv_b_ptr, gn_w_ptr,
    sum_cb_g_ptr, sum_cb_sq_g_ptr, sum_gn_w_g_ptr,
    sum_gn_b_g_ptr, sum_gn_w_cb_g_ptr,
    out_ptr,
    OC: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    CHANS_PER_GROUP: tl.constexpr,
    SPATIAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps: tl.constexpr,
    INV_TOTAL: tl.constexpr,
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

    tl.atomic_add(out_ptr + n, out_sum * INV_TOTAL)


@triton.jit
def reduce_groups_kernel(
    partial_ptr, out_ptr,
    NUM_GROUPS: tl.constexpr,
    INV_TOTAL: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    n = tl.program_id(0)
    offs = tl.arange(0, BLOCK_G)
    mask = offs < NUM_GROUPS
    v = tl.load(partial_ptr + n * NUM_GROUPS + offs, mask=mask, other=0.0)
    s = tl.sum(v, axis=0)
    tl.store(out_ptr + n, s * INV_TOTAL)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

        self._cached = False

    def _cache_params(self):
        OC = self.out_channels
        CHANS_PER_GROUP = OC // self.num_groups
        conv_b = self.conv.bias.detach().contiguous()
        gn_w = self.group_norm.weight.detach().contiguous()
        gn_b = self.group_norm.bias.detach().contiguous()
        cb_g = conv_b.view(self.num_groups, CHANS_PER_GROUP)
        gw_g = gn_w.view(self.num_groups, CHANS_PER_GROUP)
        gb_g = gn_b.view(self.num_groups, CHANS_PER_GROUP)
        self._sum_cb = cb_g.sum(dim=1).contiguous()
        self._sum_cb_sq = (cb_g * cb_g).sum(dim=1).contiguous()
        self._sum_gn_w = gw_g.sum(dim=1).contiguous()
        self._sum_gn_b = gb_g.sum(dim=1).contiguous()
        self._sum_gn_w_cb = (gw_g * cb_g).sum(dim=1).contiguous()
        self._conv_b = conv_b
        self._gn_w = gn_w
        self._cached = True

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

        if not self._cached or self._conv_b.device != x.device:
            self._cache_params()

        y = F.conv3d(x, self.conv.weight, None)
        y = y.view(N, OC, SPATIAL)

        eps = self.group_norm.eps
        inv_total = 1.0 / float(OC * SPATIAL)

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        grid = (N, self.num_groups)
        gn_mean_kernel[grid](
            y, self._conv_b, self._gn_w,
            self._sum_cb, self._sum_cb_sq, self._sum_gn_w,
            self._sum_gn_b, self._sum_gn_w_cb,
            out,
            OC,
            self.num_groups,
            CHANS_PER_GROUP,
            SPATIAL,
            GROUP_SIZE,
            eps,
            inv_total,
        )

        return out