import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_stats_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, stats_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    G, CPG,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # program ids: (n, oc, s_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_start = pid_s * BLOCK_S
    s_offs = s_start + tl.arange(0, BLOCK_S)
    S = OD * OH * OW
    s_mask = s_offs < S

    od = s_offs // (OH * OW)
    rem = s_offs - od * (OH * OW)
    oh = rem // OW
    ow = rem - oh * OW

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Loop over input channels and kernel dims
    for ic in range(0, IC):
        for kd in range(0, KD):
            for kh in range(0, KH):
                for kw in range(0, KW):
                    id_ = od + kd
                    ih = oh + kh
                    iw = ow + kw
                    x_idx = ((pid_n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
                    w_idx = ((pid_oc * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                    xv = tl.load(x_ptr + x_idx, mask=s_mask, other=0.0)
                    wv = tl.load(w_ptr + w_idx)
                    acc += xv * wv

    bv = tl.load(b_ptr + pid_oc)
    acc = acc + bv

    # store output
    y_idx = ((pid_n * OC + pid_oc) * OD * OH * OW) + s_offs
    tl.store(y_ptr + y_idx, acc, mask=s_mask)

    # accumulate group stats
    g = pid_oc // CPG
    acc_masked = tl.where(s_mask, acc, 0.0)
    sum_v = tl.sum(acc_masked, axis=0)
    sum_sq = tl.sum(acc_masked * acc_masked, axis=0)
    stats_base = (pid_n * G + g) * 2
    tl.atomic_add(stats_ptr + stats_base, sum_v)
    tl.atomic_add(stats_ptr + stats_base + 1, sum_sq)


@triton.jit
def gn_mean_kernel(
    y_ptr, stats_ptr, w_ptr, b_ptr, out_ptr,
    N, G, CPG, S,
    eps: tl.constexpr,
    inv_total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # one program per (n, g)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    group_size = CPG * S

    s_v = tl.load(stats_ptr + (n * G + g) * 2)
    s_sq = tl.load(stats_ptr + (n * G + g) * 2 + 1)
    inv_gs = 1.0 / group_size
    mean = s_v * inv_gs
    var = s_sq * inv_gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    base = n * (G * CPG * S) + g * CPG * S
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(y_ptr + base + idx, mask=mask, other=0.0)
        c_local = idx // S
        c_global = g * CPG + c_local
        w = tl.load(w_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(b_ptr + c_global, mask=mask, other=0.0)
        y = (v - mean) * rstd * w + b
        y = tl.where(mask, y, 0.0)
        acc += y
    group_sum = tl.sum(acc, axis=0)
    tl.atomic_add(out_ptr + n, group_sum * inv_total)


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
        S = OD * OH * OW
        G = self.num_groups
        CPG = OC // G

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        y = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)
        stats = torch.zeros((N, G, 2), device=x.device, dtype=torch.float32)

        BLOCK_S = 256
        grid = (N, OC, (S + BLOCK_S - 1) // BLOCK_S)
        conv3d_stats_kernel[grid](
            x, weight, bias, y, stats,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            G, CPG,
            KD, KH, KW,
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2,
        )

        total = OC * S
        inv_total = 1.0 / total
        out = torch.zeros(N, device=x.device, dtype=x.dtype)
        BLOCK = 2048
        gn_mean_kernel[(N * G,)](
            y, stats, self.group_norm.weight, self.group_norm.bias, out,
            N, G, CPG, S,
            self.eps, inv_total,
            BLOCK=BLOCK, num_warps=8, num_stages=3,
        )
        return out