import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC,
    IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    SUB: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    oh = sp_offs // OW
    ow = sp_offs % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    x_base = pid_n * (IC * IH * IW)
    K = IC * KH * KW

    # Loop unified over IC*KH*KW
    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh
                iw = ow + kw
                x_off = x_base + ic * (IH * IW) + ih * IW + iw
                x_vals = tl.load(x_ptr + x_off, mask=sp_mask, other=0.0)

                w_off = oc_offs * K + ic * (KH * KW) + kh * KW + kw
                w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                acc += w_vals[:, None] * x_vals[None, :]

    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_vals[:, None] - SUB

    # Mish via stable softplus
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    out = acc * tanh_sp

    out_off = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=out_mask)


@triton.jit
def conv2d_mish_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC,
    IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    SUB: tl.constexpr,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
    BLOCK_K: tl.constexpr,  # K tile = IC*KH*KW (or fits)
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_sp = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_sp = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < OC
    sp_mask = offs_sp < (OH * OW)

    oh = offs_sp // OW
    ow = offs_sp % OW

    K_total = IC * KH * KW
    HW = IH * IW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_base = pid_n * IC * HW

    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_total

        # decompose k -> ic, kh, kw
        ic = offs_k // (KH * KW)
        rem = offs_k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # weight: [BLOCK_M, BLOCK_K]
        w_off = offs_m[:, None] * K_total + offs_k[None, :]
        w_mask = m_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        # input: [BLOCK_K, BLOCK_N]
        ih = oh[None, :] + kh[:, None]  # [BLOCK_K, BLOCK_N]
        iw = ow[None, :] + kw[:, None]
        x_off = x_base + ic[:, None] * HW + ih * IW + iw
        x_mask = k_mask[:, None] & sp_mask[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        acc += tl.dot(w_tile, x_tile, allow_tf32=True)

    b_vals = tl.load(b_ptr + offs_m, mask=m_mask, other=0.0)
    acc = acc + b_vals[:, None] - SUB

    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    out = acc * tanh_sp

    out_off = pid_n * (OC * OH * OW) + offs_m[:, None] * (OH * OW) + offs_sp[None, :]
    out_mask = m_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = w.shape[0]
        KH = w.shape[2]
        KW = w.shape[3]
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        SUB = float(self.subtract_value_1 + self.subtract_value_2)

        K_total = IC * KH * KW  # for typical: 8*3*3=72

        # Use GEMM-style implicit conv
        BLOCK_M = 64
        BLOCK_N = 64
        # round K up to next pow2 >= K_total
        BLOCK_K = 1
        while BLOCK_K < K_total:
            BLOCK_K *= 2
        BLOCK_K = min(BLOCK_K, 128)

        grid = (N, triton.cdiv(OC, BLOCK_M), triton.cdiv(OH * OW, BLOCK_N))

        conv2d_mish_gemm_kernel[grid](
            x, w, b, out,
            N, IC,
            IH, IW,
            OC,
            OH, OW,
            KH, KW,
            SUB,
            BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=4, num_stages=3,
        )
        return out