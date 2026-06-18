import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=3),
    ],
    key=['SPATIAL', 'CHANS_PER_GROUP'],
)
@triton.jit
def gn_mean_kernel(
    y_ptr, conv_b_ptr, gn_w_ptr, gn_b_ptr, partial_ptr,
    OC: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    CHANS_PER_GROUP: tl.constexpr,
    SPATIAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps: tl.constexpr,
    INV_TOTAL: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # one program per (batch, group)
    n = tl.program_id(0)
    g = tl.program_id(1)

    oc_start = g * CHANS_PER_GROUP
    # base ptr: y[n, oc_start, 0]
    base = n * OC * SPATIAL + oc_start * SPATIAL

    s_offs = tl.arange(0, BLOCK_S)

    # Phase 1: compute sum and sum_sq over the group (fusing conv bias add)
    sum_val = 0.0
    sum_sq = 0.0

    for c_idx in tl.static_range(CHANS_PER_GROUP):
        ch_base = base + c_idx * SPATIAL
        cb = tl.load(conv_b_ptr + oc_start + c_idx)
        for s_start in range(0, SPATIAL, BLOCK_S):
            s = s_start + s_offs
            s_mask = s < SPATIAL
            v = tl.load(y_ptr + ch_base + s, mask=s_mask, other=0.0) + cb
            v = tl.where(s_mask, v, 0.0)
            sum_val += tl.sum(v, axis=0)
            sum_sq += tl.sum(v * v, axis=0)

    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Phase 2: normalize and accumulate output mean
    out_sum = 0.0
    for c_idx in tl.static_range(CHANS_PER_GROUP):
        ch_base = base + c_idx * SPATIAL
        cb = tl.load(conv_b_ptr + oc_start + c_idx)
        gn_w = tl.load(gn_w_ptr + oc_start + c_idx)
        gn_b = tl.load(gn_b_ptr + oc_start + c_idx)
        scale = inv_std * gn_w
        shift = gn_b + (cb - mean) * scale
        for s_start in range(0, SPATIAL, BLOCK_S):
            s = s_start + s_offs
            s_mask = s < SPATIAL
            v = tl.load(y_ptr + ch_base + s, mask=s_mask, other=0.0)
            normed = v * scale + shift
            normed_masked = tl.where(s_mask, normed, 0.0)
            out_sum += tl.sum(normed_masked, axis=0)

    result = out_sum * INV_TOTAL
    tl.store(partial_ptr + n * NUM_GROUPS + g, result)


@triton.jit
def reduce_groups_kernel(
    partial_ptr, out_ptr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    n = tl.program_id(0)
    offs = tl.arange(0, BLOCK_G)
    mask = offs < NUM_GROUPS
    v = tl.load(partial_ptr + n * NUM_GROUPS + offs, mask=mask, other=0.0)
    s = tl.sum(v, axis=0)
    tl.store(out_ptr + n, s)


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

        # Use cuDNN-backed conv3d for the heavy lifting; fuse bias add into GN kernel
        y = F.conv3d(x, self.conv.weight, None)
        y = y.contiguous().view(N, OC, SPATIAL)

        conv_b = self.conv.bias.contiguous()
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps
        inv_total = 1.0 / float(OC * SPATIAL)

        partial = torch.empty((N, self.num_groups), device=x.device, dtype=torch.float32)
        out = torch.empty(N, device=x.device, dtype=torch.float32)

        grid = (N, self.num_groups)
        gn_mean_kernel[grid](
            y, conv_b, gn_w, gn_b, partial,
            OC,
            self.num_groups,
            CHANS_PER_GROUP,
            SPATIAL,
            GROUP_SIZE,
            eps,
            inv_total,
        )

        # Reduce groups -> out
        BLOCK_G = triton.next_power_of_2(self.num_groups)
        reduce_groups_kernel[(N,)](
            partial, out,
            self.num_groups,
            BLOCK_G,
            num_warps=1,
        )

        return out