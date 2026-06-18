import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gn_mean_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, G, CPG, S, C,
    eps,
    inv_total,
    BLOCK_S: tl.constexpr,
):
    # one program per (n, g)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_size = CPG * S
    base = n * (C * S) + g * (CPG * S)

    sum_val = 0.0
    sum_sq = 0.0
    num_blocks = (group_size + BLOCK_S - 1) // BLOCK_S
    for b in range(0, num_blocks):
        offs = b * BLOCK_S + tl.arange(0, BLOCK_S)
        mask = offs < group_size
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        sum_val += tl.sum(v, axis=0)
        sum_sq += tl.sum(v * v, axis=0)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # second pass: compute (v - mean) * rstd * w_c + b_c, sum, then atomically add
    # contribution to out[n] scaled by 1/total.
    partial = 0.0
    for b in range(0, num_blocks):
        offs = b * BLOCK_S + tl.arange(0, BLOCK_S)
        mask = offs < group_size
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        c_in_group = offs // S
        c = g * CPG + c_in_group
        w = tl.load(weight_ptr + c, mask=mask, other=0.0)
        bi = tl.load(bias_ptr + c, mask=mask, other=0.0)
        out_val = (v - mean) * rstd * w + bi
        # zero out masked
        out_val = tl.where(mask, out_val, 0.0)
        partial += tl.sum(out_val, axis=0)

    # accumulate into out[n]
    tl.atomic_add(out_ptr + n, partial * inv_total)


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

        out = torch.zeros(N, device=x.device, dtype=x.dtype)

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

        grid = (N * G,)
        fused_gn_mean_kernel[grid](
            x, weight, bias, out,
            N, G, CPG, S, C,
            self.eps,
            inv_total,
            BLOCK_S=BLOCK_S,
            num_warps=num_warps,
        )
        return out