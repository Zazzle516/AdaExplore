import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_implicit_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    inv_div,
    BLOCK_M: tl.constexpr,  # spatial output tile per program
    BLOCK_N: tl.constexpr,  # OC tile (== OC here)
    BLOCK_K: tl.constexpr,  # K-axis tile
    K_TOTAL: tl.constexpr,  # IC*KD*KH*KW
):
    pid_n = tl.program_id(0)        # batch
    pid_m = tl.program_id(1)        # output spatial tile

    M_TOTAL = OD * OH * OW
    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M_TOTAL

    # decompose m_offs into (od, oh, ow)
    od = m_offs // (OH * OW)
    rem = m_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    n_offs = tl.arange(0, BLOCK_N)  # OC indices
    n_mask = n_offs < OC

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # K loop
    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_TOTAL

        # decompose k into (ic, kd, kh, kw)
        ic = k_offs // (KD * KH * KW)
        krem = k_offs % (KD * KH * KW)
        kd = krem // (KH * KW)
        krem2 = krem % (KH * KW)
        kh = krem2 // KW
        kw = krem2 % KW

        # Load weight tile [BLOCK_N, BLOCK_K]: w[oc, ic, kd, kh, kw]
        w_idx = ((n_offs[:, None] * IC + ic[None, :]) * KD + kd[None, :]) * KH * KW + kh[None, :] * KW + kw[None, :]
        w_m = n_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_idx, mask=w_m, other=0.0)

        # Load input tile [BLOCK_M, BLOCK_K]: x[n, ic, od+kd, oh+kh, ow+kw]
        id_ = od[:, None] + kd[None, :]
        ih_ = oh[:, None] + kh[None, :]
        iw_ = ow[:, None] + kw[None, :]
        x_idx = ((pid_n * IC + ic[None, :]) * ID + id_) * IH * IW + ih_ * IW + iw_
        x_m = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_idx, mask=x_m, other=0.0)

        # GEMM: [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(x_tile, tl.trans(w_tile))

    # Add bias (per-OC), apply divisor
    b_vals = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0)
    acc = (acc + b_vals[None, :]) * inv_div

    # Store: out[n, oc, od, oh, ow]
    out_idx = ((pid_n * OC + n_offs[None, :]) * OD * OH * OW) + m_offs[:, None]
    out_m = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_m)


@triton.jit
def fused_pool_avg_bias_sum_kernel(
    conv_ptr, bias_ptr, out_ptr,
    N, OC, OD, OH, OW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
    POOL_VOL: tl.constexpr,
):
    n = tl.program_id(0)
    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Accumulator over pooled positions -> per channel sum
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    spatial = OD * OH * OW
    base = n * OC * spatial

    for pd in range(0, PD):
        for ph in range(0, PH):
            for pw in range(0, PW):
                max_val = tl.full([BLOCK_OC], -1e38, dtype=tl.float32)
                for dd in range(0, 2):
                    for dh in range(0, 2):
                        for dw in range(0, 2):
                            od = pd * 2 + dd
                            oh = ph * 2 + dh
                            ow = pw * 2 + dw
                            idx = base + oc_offs * spatial + od * OH * OW + oh * OW + ow
                            v = tl.load(conv_ptr + idx, mask=oc_mask, other=-1e38)
                            max_val = tl.maximum(max_val, v)
                acc += max_val

    pool_count = PD * PH * PW
    avg = acc / pool_count

    bias_vals = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    avg = avg + bias_vals

    avg_masked = tl.where(oc_mask, avg, 0.0)
    s = tl.sum(avg_masked, axis=0)
    tl.store(out_ptr + n, s)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.KD = self.KH = self.KW = kernel_size
        else:
            self.KD, self.KH, self.KW = kernel_size
        if isinstance(pool_size, int):
            self.PD = self.PH = self.PW = pool_size
        else:
            self.PD, self.PH, self.PW = pool_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD, KH, KW = self.KD, self.KH, self.KW
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        weight = self.conv.weight.contiguous()
        cbias = self.conv.bias.contiguous()

        conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

        K_TOTAL = IC * KD * KH * KW
        # Pad K_TOTAL up to a power-of-2 friendly BLOCK_K
        BLOCK_M = 64
        BLOCK_N = 16  # OC=16
        # Choose BLOCK_K
        BLOCK_K = 32
        # Ensure BLOCK_N >= 16 for tl.dot
        M_TOTAL = OD * OH * OW
        grid = (N, triton.cdiv(M_TOTAL, BLOCK_M))

        conv3d_implicit_gemm_kernel[grid](
            x, weight, cbias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            1.0 / self.divisor,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            K_TOTAL=K_TOTAL,
            num_warps=4, num_stages=2,
        )

        # Pool + avg + bias + sum
        PD = OD // self.PD
        PH = OH // self.PH
        PW = OW // self.PW

        out = torch.empty((N,), device=x.device, dtype=torch.float32)
        bias_flat = self.bias.view(-1).contiguous()

        BLOCK_OC = 16  # next pow2 >= OC
        fused_pool_avg_bias_sum_kernel[(N,)](
            conv_out, bias_flat, out,
            N, OC, OD, OH, OW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC,
            POOL_VOL=8,
            num_warps=2, num_stages=2,
        )

        # Output shape: input was [N, OC, 1, 1, 1] then sum over dim=1 -> [N, 1, 1, 1]
        return out.view(N, 1, 1, 1)