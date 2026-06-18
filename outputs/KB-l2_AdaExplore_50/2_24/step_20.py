import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=3),
    ],
    key=['N', 'C_in', 'D', 'H', 'W', 'C_out', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv3d_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in: tl.constexpr, D, H, W,
    C_out: tl.constexpr, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N, cdiv(OH*OW, BLOCK_HW))
    n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    hw_start = pid_hw * BLOCK_HW
    hw_offs = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    c_offs = tl.arange(0, BLOCK_C)
    mask_c = c_offs < C_out

    # Load bias (BLOCK_C,)
    bias = tl.load(b_ptr + c_offs, mask=mask_c, other=0.0)

    # Precompute K=C_in*KD*KH*KW indices for x-gather and w-gather
    # k index decomposition: k = ((ic*KD + kd)*KH + kh)*KW + kw
    k_offs = tl.arange(0, K_TOTAL)  # (K,)
    kw_i = k_offs % KW
    tmp1 = k_offs // KW
    kh_i = tmp1 % KH
    tmp2 = tmp1 // KH
    kd_i = tmp2 % KD
    ic_i = tmp2 // KD

    HW = H * W
    DHW = D * HW

    # x offsets for spatial neighborhood: shape (BLOCK_HW, K)
    # base for (n, ic, kd_off + od, kh + oh, kw + ow)
    # we will add od*HW inside the od-loop
    ohw_spatial = (oh * W + ow)  # (BLOCK_HW,)

    # x_off_kspat (K,) = ic*DHW + kd*HW + kh*W + kw
    x_off_k = ic_i * DHW + kd_i * HW + kh_i * W + kw_i  # (K,)
    # x_off_nbase = n * C_in * DHW
    x_n_base = n * C_in * DHW

    # w_off (C_out, K) flattened: weight shape (C_out, C_in, KD, KH, KW), already contiguous => oc*K + k
    # We'll load weight as (BLOCK_C, K) using c_offs * K + k_offs
    w_idx = c_offs[:, None] * K_TOTAL + k_offs[None, :]  # (BLOCK_C, K)
    w_mask = mask_c[:, None] & (k_offs[None, :] < K_TOTAL)
    w_tile = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)  # (BLOCK_C, K)
    # transpose for tl.dot K dim: we want (K, BLOCK_C)
    w_t = tl.trans(w_tile)  # (K, BLOCK_C)

    # Init min over od
    min_val = tl.full((BLOCK_HW, BLOCK_C), float('inf'), dtype=tl.float32)

    for od in range(0, OD):
        # build x indices (BLOCK_HW, K)
        x_idx = x_n_base + (od * HW) + ohw_spatial[:, None] + x_off_k[None, :]
        x_mask = mask_hw[:, None] & (k_offs[None, :] < K_TOTAL)
        x_tile = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)  # (BLOCK_HW, K)

        # GEMM: acc(BLOCK_HW, BLOCK_C) = x_tile @ w_t
        acc = tl.dot(x_tile, w_t, out_dtype=tl.float32)
        acc = acc + bias[None, :]
        min_val = tl.minimum(min_val, acc)

    # Mask invalid channels to -inf
    min_val = tl.where(mask_c[None, :], min_val, -float('inf'))

    # Softmax across channels
    m = tl.max(min_val, axis=1)
    e = tl.exp(min_val - m[:, None])
    e = tl.where(mask_c[None, :], e, 0.0)
    z = tl.sum(e, axis=1)
    y = e / z[:, None]

    out_base = n * C_out * (OH * OW)
    out_idx = out_base + c_offs[None, :] * (OH * OW) + hw_offs[:, None]
    store_mask = mask_hw[:, None] & mask_c[None, :]
    tl.store(out_ptr + out_idx, y, mask=store_mask)


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

        N, C_in, D, H, W = x.shape
        C_out = self.out_channels
        K = self.kernel_size
        OD = D - K + 1
        OH = H - K + 1
        OW = W - K + 1

        if self.dim != 2:
            y = self.conv(x)
            y = torch.min(y, dim=self.dim)[0]
            return torch.softmax(y, dim=1)

        out = torch.empty((N, C_out, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_C = max(16, _next_pow2(C_out))
        K_TOTAL = _next_pow2(C_in * K * K * K)
        # Need K_TOTAL >= 16 for tl.dot
        K_TOTAL = max(16, K_TOTAL)

        grid = lambda meta: (N, triton.cdiv(OH * OW, meta['BLOCK_HW']))
        conv3d_min_softmax_kernel[grid](
            x, w, b, out,
            N, C_in, D, H, W,
            C_out, OD, OH, OW,
            K, K, K,
            K_TOTAL=K_TOTAL,
            BLOCK_C=BLOCK_C,
        )
        return out