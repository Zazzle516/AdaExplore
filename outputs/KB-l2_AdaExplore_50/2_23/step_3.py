import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mean_reduce_kernel(
    x_ptr, out_ptr,
    N, C, S,
    inv_total,
    BLOCK_S: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = (n * C + c) * S
    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(v, axis=0)
    # atomic add into out[n] * inv_total
    tl.atomic_add(out_ptr + n, acc * inv_total)


@triton.jit
def groupnorm_kernel(
    x_ptr, y_ptr,
    weight_ptr, bias_ptr,
    N, G, CPG, S,
    eps,
    BLOCK_S: tl.constexpr,
    CPG_C: tl.constexpr,
):
    # one program per (n, g), reduces over CPG*S elements
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    group_size = CPG * S

    # pointer to start of this group in x: x[n, g*CPG : (g+1)*CPG, :]
    group_base = n * (G * CPG * S) + g * CPG * S

    # compute mean and var with two passes
    sum_val = 0.0
    sumsq_val = 0.0
    for c in range(0, CPG_C):
        c_valid = c < CPG
        c_base = group_base + c * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = (offs < S) & c_valid
            v = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0)
            sum_val += tl.sum(v, axis=0)
            sumsq_val += tl.sum(v * v, axis=0)

    mean = sum_val / group_size
    var = sumsq_val / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize
    for c in range(0, CPG_C):
        c_valid = c < CPG
        c_base = group_base + c * S
        # load weight, bias for this channel
        ch_idx = g * CPG + c
        w = tl.load(weight_ptr + ch_idx, mask=c_valid, other=0.0)
        b = tl.load(bias_ptr + ch_idx, mask=c_valid, other=0.0)
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = (offs < S) & c_valid
            v = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0)
            v_norm = (v - mean) * rstd
            out = v_norm * w + b
            tl.store(y_ptr + c_base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous()
        N, C, D, H, W = x.shape
        S = D * H * W
        G = self.num_groups
        CPG = C // G

        y = torch.empty_like(x)
        # pick BLOCK_S
        BLOCK_S = 1024
        # CPG_C as power-of-two upper bound for constexpr loop bound
        CPG_C = CPG  # exact

        eps = self.group_norm.eps
        grid = (N * G,)
        groupnorm_kernel[grid](
            x, y,
            self.group_norm.weight, self.group_norm.bias,
            N, G, CPG, S,
            eps,
            BLOCK_S=BLOCK_S,
            CPG_C=CPG_C,
            num_warps=4,
        )

        # mean reduce over C, S => output shape (N,)
        out = torch.zeros(N, device=x.device, dtype=x.dtype)
        total = C * S
        inv_total = 1.0 / total
        grid2 = (N * C,)
        mean_reduce_kernel[grid2](
            y, out,
            N, C, S,
            inv_total,
            BLOCK_S=1024,
            num_warps=4,
        )
        return out