import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'K', 'OD'],
)
@triton.jit
def conv3d_min_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, D, H, W,
    OC: tl.constexpr, OD, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    K: tl.constexpr,  # IC*KD*KH*KW
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile (OH*OW)
):
    pid_n = tl.program_id(0)            # batch
    pid_m = tl.program_id(1)            # OC tile
    pid_hw = tl.program_id(2)           # spatial tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # OC indices
    offs_n = pid_hw * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial indices

    mask_m = offs_m < OC
    mask_n = offs_n < (OH * OW)

    oh = offs_n // OW
    ow = offs_n % OW

    # Precompute weight pointer base for this OC tile: W has shape (OC, K)
    # weight already laid out as (OC, IC*KD*KH*KW) contiguous
    offs_k = tl.arange(0, K)  # K is small, e.g. 81
    w_ptrs = w_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_mask = mask_m[:, None] & (offs_k[None, :] < K)
    w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # (BLOCK_M, K)

    bias = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)  # (BLOCK_M,)

    # Precompute decomposition of k = ((ic*KD + kd)*KH + kh)*KW + kw
    k_idx = tl.arange(0, K)
    kw_i = k_idx % KW
    tmp1 = k_idx // KW
    kh_i = tmp1 % KH
    tmp2 = tmp1 // KH
    kd_i = tmp2 % KD
    ic_i = tmp2 // KD

    x_n_base = pid_n * (IC * D * H * W)

    min_val = tl.full((BLOCK_M, BLOCK_N), float('inf'), dtype=tl.float32)

    for od in range(0, OD):
        # Build x tile: (K, BLOCK_N)
        # x_idx = n_base + ic*D*H*W + (od+kd)*H*W + (oh+kh)*W + (ow+kw)
        d_idx = od + kd_i  # (K,)
        # offsets per (k, n)
        # base per k:
        x_k_base = ic_i * (D * H * W) + d_idx * (H * W)  # (K,)
        # spatial per n: (oh+kh)*W + (ow+kw) per (k, n)
        # we need shape (K, BLOCK_N)
        h_pos = oh[None, :] + kh_i[:, None]  # (K, BLOCK_N)
        w_pos = ow[None, :] + kw_i[:, None]  # (K, BLOCK_N)
        x_offsets = x_n_base + x_k_base[:, None] + h_pos * W + w_pos  # (K, BLOCK_N)

        x_mask = mask_n[None, :]  # k dimension fully valid
        x_tile = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)  # (K, BLOCK_N)

        acc = tl.dot(w_tile, x_tile)  # (BLOCK_M, BLOCK_N)
        acc = acc + bias[:, None]

        min_val = tl.minimum(min_val, acc)

    # Store result -> shape (N, OC, OH*OW)
    out_base = pid_n * (OC * OH * OW) + offs_m[:, None] * (OH * OW) + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_base, min_val, mask=out_mask)


@triton.jit
def softmax_channel_kernel(
    x_ptr, out_ptr,
    N, C: tl.constexpr, S,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = tl.arange(0, BLOCK_C)
    mask_s = offs_s < S
    mask_c = offs_c < C

    base = pid_n * C * S
    ptrs = x_ptr + base + offs_c[:, None] * S + offs_s[None, :]
    mask = mask_c[:, None] & mask_s[None, :]

    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)  # (BLOCK_S,)
    e = tl.exp(x - m[None, :])
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)  # (BLOCK_S,)
    y = e / s[None, :]

    tl.store(out_ptr + base + offs_c[:, None] * S + offs_s[None, :], y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Precompute flattened weight layout (OC, IC*KD*KH*KW)
        with torch.no_grad():
            w = self.conv.weight.detach().contiguous()
            OC = w.shape[0]
            w_flat = w.view(OC, -1).contiguous()
        self.register_buffer('w_flat', w_flat, persistent=False)

    def forward(self, x):
        x = x.contiguous().cuda()
        w_flat = self.w_flat.cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, D, H, W = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1
        K = IC * KD * KH * KW

        if self.dim != 2:
            y = self.conv(x)
            y = torch.min(y, dim=self.dim)[0]
            return torch.softmax(y, dim=1)

        out = torch.empty((N, OC, OH * OW), device=x.device, dtype=torch.float32)

        # K must be power-of-2-ish for tl.arange. K=81 -> use next pow2 = 128
        K_padded = triton.next_power_of_2(K)

        grid = lambda META: (
            N,
            triton.cdiv(OC, META['BLOCK_M']),
            triton.cdiv(OH * OW, META['BLOCK_N']),
        )

        conv3d_min_gemm_kernel[grid](
            x, w_flat, b, out,
            N, IC, D, H, W,
            OC, OD, OH, OW,
            KD, KH, KW,
            K_padded,
        )

        # Softmax along channel dim
        S = OH * OW
        sm = torch.empty_like(out)
        BLOCK_C = triton.next_power_of_2(OC)
        BLOCK_S = 64
        grid2 = (N, triton.cdiv(S, BLOCK_S))
        softmax_channel_kernel[grid2](
            out, sm,
            N, OC, S,
            BLOCK_S=BLOCK_S,
            BLOCK_C=BLOCK_C,
            num_warps=2,
        )

        return sm.view(N, OC, OH, OW)