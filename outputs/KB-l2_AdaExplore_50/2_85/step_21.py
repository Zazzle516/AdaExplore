import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gn_scale_maxpool_clamp_kernel(
    x_ptr,           # input: (N, C, H, W) after conv
    gamma_ptr,       # GroupNorm weight: (C,)
    beta_ptr,        # GroupNorm bias: (C,)
    scale_ptr,       # Scale: (C,)
    out_ptr,         # output: (N, C, H//P, W//P)
    N, C, H, W,
    H_out, W_out,
    G,               # num groups
    CPG,             # channels per group
    GROUP_SIZE,      # CPG * H * W
    eps,
    clamp_min,
    clamp_max,
    P: tl.constexpr,            # maxpool kernel size
    BLOCK_SIZE: tl.constexpr,   # block for reduction
):
    # one program per (batch, group)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    # pointer to start of (n, group)
    group_start = pid_n * C * H * W + pid_g * CPG * H * W

    # compute mean and var across CPG*H*W elements
    mean_acc = 0.0
    m2_acc = 0.0

    # single-pass: sum and sum of squares
    sum_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_x2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for offset in range(0, GROUP_SIZE, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < GROUP_SIZE
        v = tl.load(x_ptr + group_start + idx, mask=mask, other=0.0)
        sum_x += tl.where(mask, v, 0.0)
        sum_x2 += tl.where(mask, v * v, 0.0)

    s = tl.sum(sum_x, axis=0)
    s2 = tl.sum(sum_x2, axis=0)

    mean = s / GROUP_SIZE
    var = s2 / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # now produce output. For each channel in group, perform normalize+scale, then maxpool over PxP windows.
    # H_out * W_out output positions per channel.
    for c_local in range(0, CPG):
        c = pid_g * CPG + c_local
        g = tl.load(gamma_ptr + c)
        b = tl.load(beta_ptr + c)
        s_scale = tl.load(scale_ptr + c)
        # combined: out = ((x - mean) * rstd * g + b) * s_scale
        # = x * (rstd * g * s_scale) + (b - mean * rstd * g) * s_scale
        a_coef = rstd * g * s_scale
        b_coef = (b - mean * rstd * g) * s_scale

        ch_start = pid_n * C * H * W + c * H * W

        # iterate over output spatial positions
        total_out = H_out * W_out
        for out_idx in range(0, total_out):
            oh = out_idx // W_out
            ow = out_idx % W_out
            ih_start = oh * P
            iw_start = ow * P

            # load P x P window
            max_val = -float('inf')
            for ph in range(0, P):
                for pw in range(0, P):
                    ih = ih_start + ph
                    iw = iw_start + pw
                    in_mask = (ih < H) & (iw < W)
                    v = tl.load(x_ptr + ch_start + ih * W + iw, mask=in_mask, other=-float('inf'))
                    val = v * a_coef + b_coef
                    max_val = tl.maximum(max_val, val)

            # clamp
            max_val = tl.minimum(tl.maximum(max_val, clamp_min), clamp_max)

            out_offset = pid_n * C * H_out * W_out + c * H_out * W_out + oh * W_out + ow
            tl.store(out_ptr + out_offset, max_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.eps = 1e-5

    def forward(self, x):
        # Run conv with torch
        x = self.conv(x)
        x = x.contiguous()
        N, C, H, W = x.shape
        P = self.maxpool_kernel_size
        H_out = H // P
        W_out = W // P

        G = self.num_groups
        CPG = C // G
        GROUP_SIZE = CPG * H * W

        out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        # pick BLOCK_SIZE
        # GROUP_SIZE could be large; pick a power-of-two block
        BLOCK_SIZE = 1024
        if GROUP_SIZE < 1024:
            BLOCK_SIZE = 256

        scale_flat = self.scale.view(-1).contiguous()

        grid = (N, G)
        fused_gn_scale_maxpool_clamp_kernel[grid](
            x, self.group_norm.weight, self.group_norm.bias, scale_flat, out,
            N, C, H, W,
            H_out, W_out,
            G, CPG, GROUP_SIZE,
            self.eps,
            self.clamp_min, self.clamp_max,
            P=P,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        return out