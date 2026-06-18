import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Conv3d kernel: implicit im2col GEMM
# One program per (N, OC_tile, output-spatial tile)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 24, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 24, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 24, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
    ],
    key=['OC', 'IC', 'KD', 'KH', 'KW', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)           # batch
    pid_m = tl.program_id(1)           # OC tile
    pid_s = tl.program_id(2)           # output spatial tile

    OUT_SPATIAL = OD * OH * OW
    K = IC * KD * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)     # OC indices
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)     # spatial indices

    mask_m = offs_m < OC
    mask_s = offs_s < OUT_SPATIAL

    # Decompose spatial index to (od, oh, ow)
    od = offs_s // (OH * OW)
    rem = offs_s - od * (OH * OW)
    oh = rem // OW
    ow = rem - oh * OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # X base for this batch
    x_n_base = pid_n * (IC * ID * IH * IW)

    # Loop over K = IC * KD * KH * KW
    for ic in range(0, IC):
        for kd in range(0, KD):
            for kh in range(0, KH):
                for kw in range(0, KW):
                    # weight index
                    k = ((ic * KD + kd) * KH + kh) * KW + kw
                    # weight shape: (OC, IC, KD, KH, KW)
                    w_off = offs_m * K + k
                    w_vals = tl.load(w_ptr + w_off, mask=mask_m, other=0.0)  # (BLOCK_M,)

                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw

                    x_off = x_n_base + ic * (ID * IH * IW) + id_ * (IH * IW) + ih_ * IW + iw_
                    x_vals = tl.load(x_ptr + x_off, mask=mask_s, other=0.0)  # (BLOCK_N,)

                    acc += w_vals[:, None] * x_vals[None, :]

    # Add bias
    b_vals = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc += b_vals[:, None]

    # Store output: (N, OC, OD, OH, OW)
    out_base = pid_n * (OC * OUT_SPATIAL)
    out_off = out_base + offs_m[:, None] * OUT_SPATIAL + offs_s[None, :]
    out_mask = mask_m[:, None] & mask_s[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# GroupNorm + mean kernel (from baseline)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 8192}, num_warps=8, num_stages=2),
    ],
    key=['group_size'],
)
@triton.jit
def group_norm_mean_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, G, CPG, S,
    group_size,
    inv_total,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    base = n * (G * CPG * S) + g * (CPG * S)

    sum_val = 0.0
    sum_sq = 0.0
    num_blocks = (group_size + BLOCK_S - 1) // BLOCK_S
    for b in range(0, num_blocks):
        offs = b * BLOCK_S + tl.arange(0, BLOCK_S)
        mask = offs < group_size
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        sum_val += tl.sum(v, axis=0)
        sum_sq += tl.sum(v * v, axis=0)

    inv_gs = 1.0 / group_size
    mean = sum_val * inv_gs
    var = sum_sq * inv_gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    acc = 0.0
    for b in range(0, num_blocks):
        offs = b * BLOCK_S + tl.arange(0, BLOCK_S)
        mask = offs < group_size
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        c_in_group = offs // S
        c = g * CPG + c_in_group
        w = tl.load(weight_ptr + c, mask=mask, other=0.0)
        bi = tl.load(bias_ptr + c, mask=mask, other=0.0)
        scale = w * rstd
        shift = bi - mean * scale
        out = v * scale + shift
        out = tl.where(mask, out, 0.0)
        acc += tl.sum(out, axis=0)

    tl.atomic_add(out_ptr + n, acc * inv_total)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = 1e-5

    def forward(self, x):
        x = x.cuda().contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        weight = self.conv.weight.contiguous()  # (OC, IC, KD, KH, KW)
        bias = self.conv.bias.contiguous()      # (OC,)

        conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        OUT_SPATIAL = OD * OH * OW

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(OUT_SPATIAL, meta['BLOCK_N']),
        )

        conv3d_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
        )

        # GroupNorm + mean
        S = OD * OH * OW
        G = self.num_groups
        CPG = OC // G
        group_size = CPG * S
        total = OC * S
        inv_total = 1.0 / total

        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()

        out = torch.zeros(N, device=x.device, dtype=x.dtype)

        grid_gn = (N * G,)
        group_norm_mean_kernel[grid_gn](
            conv_out, gn_w, gn_b, out,
            N, G, CPG, S,
            group_size,
            inv_total,
            self.eps,
        )
        return out