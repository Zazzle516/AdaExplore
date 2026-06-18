import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=3),
    ],
    key=['OH', 'OW', 'OC', 'IC', 'KD'],
)
@triton.jit
def conv3d_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, D, H, W,
    OC: tl.constexpr, OD: tl.constexpr, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,  # padded IC*KD*KH*KW
    BLOCK_HW: tl.constexpr,
):
    # grid: (N, ceil(OH*OW / BLOCK_HW))
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]
    s_total = OH * OW
    mask_s = offs < s_total
    oh = offs // OW
    ow = offs % OW

    oc_offs = tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    mask_oc = oc_offs < OC

    # K dimension: IC*KD*KH*KW
    k_offs = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    K_TOTAL = IC * KD * KH * KW
    mask_k = k_offs < K_TOTAL

    # Decompose k -> (ic, kd, kh, kw) for input gather
    KDHW = KD * KH * KW
    KHW = KH * KW
    ic_k = k_offs // KDHW
    rem = k_offs % KDHW
    kd_k = rem // KHW
    rem2 = rem % KHW
    kh_k = rem2 // KW
    kw_k = rem2 % KW

    bias = tl.load(b_ptr + oc_offs, mask=mask_oc, other=0.0)  # [BLOCK_OC]

    # Weight is pre-reshaped to (OC, K_TOTAL) contiguous
    w_off = oc_offs[:, None] * K_TOTAL + k_offs[None, :]
    w_mask = mask_oc[:, None] & mask_k[None, :]

    INF = float('inf')
    min_val = tl.full((BLOCK_OC, BLOCK_HW), INF, dtype=tl.float32)

    # Precompute static input-offset components
    base_n = pid_n * IC * D * H * W
    # x offset depends on (od+kd), so we add od*H*W later separately by adjusting kd term:
    # For a fixed od, x_off = base_n + ic*(D*H*W) + (od+kd)*H*W + kh*W + kw + (oh*W+ow)
    # Precompute the part independent of od: ic*(D*H*W) + kd*(H*W) + kh*W + kw
    x_k_part = ic_k * (D * H * W) + kd_k * (H * W) + kh_k * W + kw_k  # [BLOCK_K]
    hw_part = oh * W + ow  # [BLOCK_HW]

    x_mask = mask_k[:, None] & mask_s[None, :]

    # Load weight once (constant across od loop)
    w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_OC, BLOCK_K]

    for od in range(0, OD):
        x_off = (base_n + od * (H * W)
                 + x_k_part[:, None]
                 + hw_part[None, :])
        x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_K, BLOCK_HW]
        acc = tl.dot(w_tile, x_tile, allow_tf32=True)  # [BLOCK_OC, BLOCK_HW]
        acc = acc + bias[:, None]
        min_val = tl.minimum(min_val, acc)

    # softmax over OC dimension (axis=0)
    min_val = tl.where(mask_oc[:, None], min_val, -INF)
    m = tl.max(min_val, axis=0)  # [BLOCK_HW]
    e = tl.exp(min_val - m[None, :])
    e = tl.where(mask_oc[:, None], e, 0.0)
    s = tl.sum(e, axis=0)  # [BLOCK_HW]
    out_v = e / s[None, :]

    out_off = (pid_n * OC + oc_offs[:, None]) * s_total + offs[None, :]
    store_mask = mask_oc[:, None] & mask_s[None, :]
    tl.store(out_ptr + out_off, out_v, mask=store_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, D, H, W = x.shape
        OC, _, KD, KH, KW = w.shape

        if self.dim == 2:
            OD = D - KD + 1
            OH = H - KH + 1
            OW = W - KW + 1

            out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)
            BLOCK_OC = max(16, _next_pow2(OC))
            K_TOTAL = IC * KD * KH * KW
            BLOCK_K = max(16, _next_pow2(K_TOTAL))

            # Reshape weight to (OC, K_TOTAL) contiguous
            w_flat = w.reshape(OC, K_TOTAL).contiguous()

            grid = lambda meta: (N, (OH * OW + meta['BLOCK_HW'] - 1) // meta['BLOCK_HW'])
            conv3d_min_softmax_kernel[grid](
                x, w_flat, b, out,
                N, IC, D, H, W,
                OC, OD, OH, OW,
                KD, KH, KW,
                BLOCK_OC=BLOCK_OC,
                BLOCK_K=BLOCK_K,
            )
            return out
        else:
            y = self.conv(x)
            y = torch.min(y, dim=self.dim)[0]
            y = torch.softmax(y, dim=1)
            return y