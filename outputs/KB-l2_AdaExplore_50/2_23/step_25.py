import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    BLOCK_N: tl.constexpr,
):
    # grid: (N, OC, OD * OH * OW / BLOCK_N) - we use one program per (n, oc, spatial-tile)
    pid_n = tl.program_id(0)  # batch
    pid_oc = tl.program_id(1)  # output channel
    pid_s = tl.program_id(2)  # spatial tile

    spatial_size = OD * OH * OW
    s_offsets = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)
    s_mask = s_offsets < spatial_size

    od = s_offsets // (OH * OW)
    rem = s_offsets % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over IC, KD, KH, KW
    for ic in range(0, IC):
        for kd in range(0, KD):
            for kh in range(0, KH):
                for kw in range(0, KW):
                    id_ = od + kd
                    ih = oh + kh
                    iw = ow + kw
                    x_idx = ((pid_n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
                    w_idx = ((pid_oc * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                    x_val = tl.load(x_ptr + x_idx, mask=s_mask, other=0.0)
                    w_val = tl.load(w_ptr + w_idx)
                    acc += x_val * w_val

    b_val = tl.load(b_ptr + pid_oc)
    acc += b_val

    out_idx = ((pid_n * OC + pid_oc) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_idx, acc, mask=s_mask)


@triton.jit
def gn_mean_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, G, CPG, S,
    eps,
    inv_total,
    BLOCK: tl.constexpr,
):
    # one program per (n, g) - computes GN for the group, accumulates partial sum for batch mean
    # then atomic add to out_ptr[n]
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    group_size = CPG * S
    base = n * (G * CPG * S) + g * CPG * S

    sum_v = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)

    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_v += v
        sum_sq += v * v

    s_sum = tl.sum(sum_v, axis=0)
    s_sq = tl.sum(sum_sq, axis=0)
    mean = s_sum / group_size
    var = s_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Now compute normalized * w + b, sum across group, accumulate to out[n]
    partial = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        c_local = idx // S
        c_global = g * CPG + c_local
        w = tl.load(w_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(b_ptr + c_global, mask=mask, other=0.0)
        y = (v - mean) * rstd * w + b
        partial += tl.where(mask, y, 0.0)

    group_total = tl.sum(partial, axis=0)
    contribution = group_total * inv_total
    tl.atomic_add(out_ptr + n, contribution)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        BLOCK_N = 128
        spatial_size = OD * OH * OW
        grid = (N, OC, (spatial_size + BLOCK_N - 1) // BLOCK_N)
        conv3d_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            BLOCK_N=BLOCK_N, num_warps=4,
        )

        # GroupNorm + mean reduction fused
        S = OD * OH * OW
        G = self.num_groups
        CPG = OC // G
        total = OC * S
        inv_total = 1.0 / total

        out = torch.zeros(N, device=x.device, dtype=x.dtype)
        BLOCK = 1024
        gn_mean_kernel[(N * G,)](
            conv_out, self.group_norm.weight, self.group_norm.bias, out,
            N, G, CPG, S,
            self.eps, inv_total,
            BLOCK=BLOCK, num_warps=4,
        )

        return out