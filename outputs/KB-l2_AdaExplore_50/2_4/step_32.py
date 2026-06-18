import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_mish_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids: (n, oc_tile, sp_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    # M = OH*OW (spatial), N = OC
    sp_start = pid_sp * BLOCK_M
    oc_start = pid_oc * BLOCK_N

    offs_m = sp_start + tl.arange(0, BLOCK_M)  # spatial indices
    offs_n = oc_start + tl.arange(0, BLOCK_N)  # oc indices

    # decode oh, ow from offs_m
    oh = offs_m // OW
    ow = offs_m % OW

    m_mask = offs_m < (OH * OW)
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K = IC * KH * KW
    # input batch base
    x_batch_ptr = x_ptr + pid_n * IC * H * W

    offs_k = tl.arange(0, BLOCK_K)
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K
        ic = k_idx // (KH * KW)
        kh = (k_idx % (KH * KW)) // KW
        kw = k_idx % KW

        # Input load: x[n, ic, oh+kh, ow+kw]
        # shape [BLOCK_M, BLOCK_K]
        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw[None, :]
        ic_b = ic[None, :]
        x_off = ic_b * (H * W) + ih * W + iw
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_batch_ptr + x_off, mask=x_mask, other=0.0)

        # Weight load: w[oc, ic, kh, kw] -> shape [BLOCK_K, BLOCK_N]
        # weight layout (OC, IC*KH*KW), access w[oc, k_idx]
        w_off = offs_n[None, :] * K + k_idx[:, None]
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # Add bias
    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b_vals[None, :]

    # Double mish: y = x * tanh(softplus(x)) with numerically stable softplus
    # softplus(x) = max(x,0) + log1p(exp(-|x|))
    abs_acc = tl.abs(acc)
    sp1 = tl.maximum(acc, 0.0) + tl.log(1.0 + tl.exp(-abs_acc))
    t1 = 2.0 * tl.sigmoid(2.0 * sp1) - 1.0
    y = acc * t1

    abs_y = tl.abs(y)
    sp2 = tl.maximum(y, 0.0) + tl.log(1.0 + tl.exp(-abs_y))
    t2 = 2.0 * tl.sigmoid(2.0 * sp2) - 1.0
    z = y * t2

    # Store: out[n, oc, oh, ow]
    out_batch_ptr = out_ptr + pid_n * OC * OH * OW
    out_off = offs_n[None, :] * (OH * OW) + offs_m[:, None]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_batch_ptr + out_off, z, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        weight = self.conv.weight.contiguous().view(OC, IC * KH * KW).contiguous()
        bias = self.conv.bias.contiguous()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        M = OH * OW
        grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_N']), triton.cdiv(M, META['BLOCK_M']))

        conv_mish_mish_kernel[grid](
            x, weight, bias, out,
            N, IC, H, W,
            OC, OH, OW,
            KH, KW,
        )
        return out