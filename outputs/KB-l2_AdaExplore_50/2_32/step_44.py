import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['N', 'OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_scale_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC: tl.constexpr, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one batch n and a tile of output spatial of size BLOCK_M.
    # Materializes the FULL OC vector per output pixel in registers, computes the
    # full conv via a single K-loop, scales, then min-reduces over OC inline.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = OH * OW
    K = IC * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    oh = offs_m // OW
    ow = offs_m % OW

    offs_oc = tl.arange(0, OC)  # full OC

    acc = tl.zeros((BLOCK_M, OC), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        ic = offs_k // (KH * KW)
        khw = offs_k % (KH * KW)
        kh = khw // KW
        kw = khw % KW

        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw[None, :]
        ic_b = tl.broadcast_to(ic[None, :], (BLOCK_M, BLOCK_K))

        x_offset = (
            pid_n * IC * IH * IW
            + ic_b * IH * IW
            + ih * IW
            + iw
        )
        x_valid = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_offset, mask=x_valid, other=0.0)

        # weight: [OC, IC*KH*KW] - shape [OC, BLOCK_K]
        w_offset = offs_oc[:, None] * (IC * KH * KW) + offs_k[None, :]
        w_valid = k_mask[None, :]
        w_vals = tl.load(w_ptr + w_offset, mask=w_valid, other=0.0)

        # acc[BLOCK_M, OC] += x[BLOCK_M, BLOCK_K] @ w.T[BLOCK_K, OC]
        acc += tl.dot(x_vals, tl.trans(w_vals))

    bias = tl.load(b_ptr + offs_oc)
    acc = acc + bias[None, :]
    acc = acc * scale

    # min over OC axis
    out_min = tl.min(acc, axis=1)

    out_offset = pid_n * OH * OW + offs_m
    tl.store(out_ptr + out_offset, out_min, mask=m_mask)


def conv2d_scale_min(x, weight, bias, scale):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    out = torch.empty((N, 1, OH, OW), device=x.device, dtype=torch.float32)

    M = OH * OW
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), N)

    conv2d_scale_min_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        float(scale),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.cuda().contiguous()
        b = self.conv.bias.cuda().contiguous()
        return conv2d_scale_min(x, w, b, self.scale_factor)