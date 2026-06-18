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
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, sum_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KT, KH, KW,
    M, K,
    NEG_SLOPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    OHW = OH * OW
    ODHW = OD * OHW
    n_idx = offs_m // ODHW
    rem = offs_m % ODHW
    od_idx = rem // OHW
    rem2 = rem % OHW
    oh_idx = rem2 // OW
    ow_idx = rem2 % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHW = KH * KW
    KTHW = KT * KHW
    IDHIW = ID * IH * IW
    IHIW = IH * IW

    # base input offset (n, ic=0, id=od, ih=oh, iw=ow)
    base_in = (n_idx * IC * IDHIW
               + od_idx * IHIW
               + oh_idx * IW
               + ow_idx)  # [BLOCK_M]

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        k_mask = k_idx < K

        ic = k_idx // KTHW
        krem = k_idx % KTHW
        kt = krem // KHW
        krem2 = krem % KHW
        kh = krem2 // KW
        kw = krem2 % KW

        # delta for each k: ic*IDHIW + kt*IHIW + kh*IW + kw
        k_delta = ic * IDHIW + kt * IHIW + kh * IW + kw  # [BLOCK_K]

        x_off = base_in[:, None] + k_delta[None, :]
        x_load_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_load_mask, other=0.0)

        w_off = offs_n[None, :] * K + k_idx[:, None]
        w_load_mask = k_mask[:, None] & (offs_n[None, :] < OC)
        w_tile = tl.load(w_ptr + w_off, mask=w_load_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    n_mask = offs_n < OC
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    acc = tl.where(acc >= 0, acc, acc * NEG_SLOPE)

    s = tl.load(sum_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + s[None, :]

    acc = tl.maximum(acc, -1.0)
    acc = tl.minimum(acc, 1.0)

    inv_sqrt2 = 0.70710678118654752440
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

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