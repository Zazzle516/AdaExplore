import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_gn_mean_kernel(
    x_ptr, w_ptr, b_ptr, gn_w_ptr, gn_b_ptr, out_ptr,
    N, IC: tl.constexpr, OC: tl.constexpr,
    D_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    D_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    K: tl.constexpr, NUM_GROUPS: tl.constexpr,
    CHANS_PER_GROUP: tl.constexpr,
    SPATIAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # one program per (batch, group)
    n = tl.program_id(0)
    g = tl.program_id(1)

    sum_val = 0.0
    sum_sq = 0.0

    s_offs = tl.arange(0, BLOCK_S)
    oc_start = g * CHANS_PER_GROUP

    # Phase 1: conv + accumulate sum, sum_sq for the group
    for c_idx in tl.static_range(CHANS_PER_GROUP):
        oc = oc_start + c_idx
        bias_val = tl.load(b_ptr + oc)

        for s_start in range(0, SPATIAL, BLOCK_S):
            s = s_start + s_offs
            s_mask = s < SPATIAL

            ow = s % W_out
            oh = (s // W_out) % H_out
            od = s // (W_out * H_out)

            acc = tl.zeros((BLOCK_S,), dtype=tl.float32) + bias_val

            for ic in tl.static_range(IC):
                for kd in tl.static_range(K):
                    for kh in tl.static_range(K):
                        for kw in tl.static_range(K):
                            id_ = od + kd
                            ih = oh + kh
                            iw = ow + kw

                            x_off = (((n * IC + ic) * D_in + id_) * H_in + ih) * W_in + iw
                            w_off = (((oc * IC + ic) * K + kd) * K + kh) * K + kw

                            x_val = tl.load(x_ptr + x_off, mask=s_mask, other=0.0)
                            w_val = tl.load(w_ptr + w_off)
                            acc += x_val * w_val

            acc_masked = tl.where(s_mask, acc, 0.0)
            sum_val += tl.sum(acc_masked, axis=0)
            sum_sq += tl.sum(acc_masked * acc_masked, axis=0)

    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Phase 2: recompute conv, normalize, accumulate output mean
    out_sum = 0.0

    for c_idx in tl.static_range(CHANS_PER_GROUP):
        oc = oc_start + c_idx
        bias_val = tl.load(b_ptr + oc)
        gn_w = tl.load(gn_w_ptr + oc)
        gn_b = tl.load(gn_b_ptr + oc)

        for s_start in range(0, SPATIAL, BLOCK_S):
            s = s_start + s_offs
            s_mask = s < SPATIAL

            ow = s % W_out
            oh = (s // W_out) % H_out
            od = s // (W_out * H_out)

            acc = tl.zeros((BLOCK_S,), dtype=tl.float32) + bias_val

            for ic in tl.static_range(IC):
                for kd in tl.static_range(K):
                    for kh in tl.static_range(K):
                        for kw in tl.static_range(K):
                            id_ = od + kd
                            ih = oh + kh
                            iw = ow + kw

                            x_off = (((n * IC + ic) * D_in + id_) * H_in + ih) * W_in + iw
                            w_off = (((oc * IC + ic) * K + kd) * K + kh) * K + kw

                            x_val = tl.load(x_ptr + x_off, mask=s_mask, other=0.0)
                            w_val = tl.load(w_ptr + w_off)
                            acc += x_val * w_val

            normed = (acc - mean) * inv_std * gn_w + gn_b
            normed_masked = tl.where(s_mask, normed, 0.0)
            out_sum += tl.sum(normed_masked, axis=0)

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
        
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps
        
        out = torch.zeros(N, device=x.device, dtype=torch.float32)
        
        BLOCK_S = 256

        grid = (N, self.num_groups)
        conv3d_gn_mean_kernel[grid](
            x, w, b, gn_w, gn_b, out,
            N, IC, OC,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            K, self.num_groups,
            CHANS_PER_GROUP,
            SPATIAL,
            GROUP_SIZE,
            eps,
            BLOCK_S,
            num_warps=8,
            num_stages=2,
        )

        return out