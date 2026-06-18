import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gn_mean_kernel(
    x_ptr, weight_ptr, bias_ptr, partial_ptr,
    N, G, CPG, S, C,
    eps: tl.constexpr,
    inv_total: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # 2D grid: (N, G)
    n = tl.program_id(0)
    g = tl.program_id(1)

    group_size = CPG * S
    base = n * (C * S) + g * (CPG * S)

    sum_v = 0.0
    sum_v2 = 0.0
    sum_wv = 0.0
    num_blocks = (group_size + BLOCK_S - 1) // BLOCK_S
    for b in range(0, num_blocks):
        offs = b * BLOCK_S + tl.arange(0, BLOCK_S)
        mask = offs < group_size
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        c_in_group = offs // S
        c = g * CPG + c_in_group
        w = tl.load(weight_ptr + c, mask=mask, other=0.0)
        wv = w * v
        # zero out masked positions for safe sums
        v_safe = tl.where(mask, v, 0.0)
        wv_safe = tl.where(mask, wv, 0.0)
        sum_v += tl.sum(v_safe, axis=0)
        sum_v2 += tl.sum(v_safe * v_safe, axis=0)
        sum_wv += tl.sum(wv_safe, axis=0)

    # per-channel sums of w and b within this group
    offs_c = tl.arange(0, BLOCK_S)
    mask_c = offs_c < CPG
    w_c = tl.load(weight_ptr + g * CPG + offs_c, mask=mask_c, other=0.0)
    b_c = tl.load(bias_ptr + g * CPG + offs_c, mask=mask_c, other=0.0)
    sum_w = tl.sum(w_c, axis=0)
    sum_b = tl.sum(b_c, axis=0)

    inv_gs = 1.0 / group_size
    mean = sum_v * inv_gs
    var = sum_v2 * inv_gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # sum over group of normalized*w + b
    partial = rstd * (sum_wv - mean * sum_w * S) + S * sum_b
    tl.store(partial_ptr + n * G + g, partial * inv_total)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.eps = 1e-5

    def forward(self, x):
        x = x.cuda()
        x = self.conv(x)
        x = x.contiguous()

        N, C, D, H, W = x.shape
        S = D * H * W
        G = self.num_groups
        CPG = C // G
        total = C * S
        inv_total = 1.0 / float(total)

        out = torch.empty(N * G, device=x.device, dtype=x.dtype)

        group_size = CPG * S
        if group_size >= 4096:
            BLOCK_S = 1024
            num_warps = 8
        elif group_size >= 1024:
            BLOCK_S = 1024
            num_warps = 4
        elif group_size >= 256:
            BLOCK_S = 512
            num_warps = 4
        else:
            BLOCK_S = 128
            num_warps = 2

        weight = self.group_norm.weight.contiguous()
        bias = self.group_norm.bias.contiguous()

        grid = (N, G)
        fused_gn_mean_kernel[grid](
            x, weight, bias, out,
            N, G, CPG, S, C,
            eps=self.eps,
            inv_total=inv_total,
            BLOCK_S=BLOCK_S,
            num_warps=num_warps,
        )
        # out currently holds partials of shape (N, G); sum over G
        return out.view(N, G).sum(dim=1)