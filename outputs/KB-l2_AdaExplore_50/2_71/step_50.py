import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv_div_lrelu_kernel(
    x_ptr,         # [N_BATCH, IC, H, W] (NCHW)
    w_ptr,         # [OC, IC, KH, KW]
    b_ptr,         # [OC]
    y_ptr,         # [N_BATCH, OC, OH, OW] (NCHW)
    M, N, K,
    N_BATCH, H, W, IC,
    OH, OW, KH, KW,
    inv_divisor,
    neg_slope,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output spatial+batch index
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC index
    offs_k = tl.arange(0, BLOCK_K)

    # decompose offs_m into (n, oh, ow)
    m_mask = offs_m < M
    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = IC * KH * KW (laid out as ic-major to match weight NCHW [OC, IC, KH, KW])
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K

        # decompose k_idx into (ic, kh, kw)
        ic = k_idx // (KH * KW)
        rem_k = k_idx % (KH * KW)
        kh = rem_k // KW
        kw = rem_k % KW

        # input addresses (NCHW): x[n, ic, oh+kh, ow+kw]
        in_h = oh_idx[:, None] + kh[None, :]
        in_w = ow_idx[:, None] + kw[None, :]
        x_off = (n_idx[:, None] * (IC * H * W)
                 + ic[None, :] * (H * W)
                 + in_h * W
                 + in_w)
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        # weight addresses (NCHW): w[oc, ic, kh, kw]
        w_off = (offs_n[:, None] * (IC * KH * KW)
                 + ic[None, :] * (KH * KW)
                 + kh[None, :] * KW
                 + kw[None, :])
        w_mask = n_mask[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)
        w_t = tl.trans(w_vals)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(x_vals, w_t)

    # bias
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # divide
    acc = acc * inv_divisor

    # leaky_relu
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    # store NCHW: y[n, oc, oh, ow]
    y_off = (n_idx[:, None] * (N * OH * OW)
             + offs_n[None, :] * (OH * OW)
             + oh_idx[:, None] * OW
             + ow_idx[:, None])
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + y_off, acc, mask=y_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # x: [N, IC, H, W] NCHW
        x = x.contiguous()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        w = self.conv.weight.detach().contiguous()  # [OC, IC, KH, KW]
        b = self.conv.bias.detach().contiguous()

        M = N * OH * OW
        N_dim = OC
        K = IC * KH * KW

        y = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N_dim, meta['BLOCK_N']),
        )

        conv_div_lrelu_kernel[grid](
            x, w, b, y,
            M, N_dim, K,
            N, H, W, IC,
            OH, OW, KH, KW,
            1.0 / float(self.divisor),
            0.01,
        )

        return y