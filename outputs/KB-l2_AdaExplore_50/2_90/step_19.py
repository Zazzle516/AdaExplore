import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, sum_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KT, KH, KW,
    M, K, # M = N*OD*OH*OW, K = IC*KT*KH*KW
    NEG_SLOPE: tl.constexpr,
    N_CONST: tl.constexpr,  # N (output N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # decompose m into (n, od, oh, ow)
    m_mask = offs_m < M
    OHW = OH * OW
    ODHW = OD * OHW
    n_idx = offs_m // ODHW
    rem = offs_m % ODHW
    od_idx = rem // OHW
    rem2 = rem % OHW
    oh_idx = rem2 // OW
    ow_idx = rem2 % OW

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop. K = IC*KT*KH*KW
    KHW = KH * KW
    KTHW = KT * KHW
    IDHIW = ID * IH * IW
    IHIW = IH * IW

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K

        # decompose k into (ic, kt, kh, kw)
        ic = k_idx // KTHW
        krem = k_idx % KTHW
        kt = krem // KHW
        krem2 = krem % KHW
        kh = krem2 // KW
        kw = krem2 % KW

        # input spatial coords for each (m, k) pair
        # id = od + kt, ih = oh + kh, iw = ow + kw  (no padding, stride=1)
        id_co = od_idx[:, None] + kt[None, :]  # [BLOCK_M, BLOCK_K]
        ih_co = oh_idx[:, None] + kh[None, :]
        iw_co = ow_idx[:, None] + kw[None, :]

        # input pointer offset
        x_off = (n_idx[:, None] * IC * IDHIW
                 + ic[None, :] * IDHIW
                 + id_co * IHIW
                 + ih_co * IW
                 + iw_co)

        x_load_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_load_mask, other=0.0)

        # weight: [OC, IC, KT, KH, KW] -> index [oc, k_idx]
        # offset = oc * (IC*KT*KH*KW) + k_idx
        w_off = offs_n[None, :] * K + k_idx[:, None]  # [BLOCK_K, BLOCK_N]
        w_load_mask = k_mask[:, None] & (offs_n[None, :] < OC)
        w_tile = tl.load(w_ptr + w_off, mask=w_load_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # bias
    n_mask = offs_n < OC
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # leaky relu
    acc = tl.where(acc >= 0, acc, acc * NEG_SLOPE)

    # add sum_tensor (per-channel)
    s = tl.load(sum_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + s[None, :]

    # clamp
    acc = tl.maximum(acc, -1.0)
    acc = tl.minimum(acc, 1.0)

    # exact GELU
    inv_sqrt2 = 0.70710678118654752440
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # store: out shape [N, OC, OD, OH, OW], offset = n*OC*ODHW + oc*ODHW + od*OHW + oh*OW + ow
    out_off = (n_idx[:, None] * OC * ODHW
               + offs_n[None, :] * ODHW
               + od_idx[:, None] * OHW
               + oh_idx[:, None] * OW
               + ow_idx[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv3d_fused(x, weight, bias, sum_tensor, neg_slope=0.2):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    sum_flat = sum_tensor.contiguous().view(-1)

    N, IC, ID, IH, IW = x.shape
    OC, _, KT, KH, KW = weight.shape
    OD = ID - KT + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    M = N * OD * OH * OW
    K = IC * KT * KH * KW

    # weight as [OC, K] - already contiguous in NCDHW so OC outer dim
    w_flat = weight.view(OC, K)

    grid = lambda meta: (
        (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
        (OC + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
    )

    conv3d_fused_kernel[grid](
        x, w_flat, bias, sum_flat, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KT, KH, KW,
        M, K,
        NEG_SLOPE=neg_slope,
        N_CONST=N,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        return conv3d_fused(x, self.conv.weight, self.conv.bias, self.sum_tensor, neg_slope=0.2)