import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
    ],
    key=['N', 'OH', 'OW', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_div_lrelu_kernel(
    x_ptr,         # NHWC: [N, H, W, IC]
    w_ptr,         # [OC, KH, KW, IC]
    b_ptr,         # [OC]
    y_ptr,         # NHWC: [N, OH, OW, OC]
    N, H, W, IC,
    OC, KH, KW,
    OH, OW,
    INV_DIV: tl.constexpr,
    NEG_SLOPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = N * OH * OW
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decompose M index into (n, oh, ow)
    m_mask = offs_m < M
    ow_idx = offs_m % OW
    oh_idx = (offs_m // OW) % OH
    n_idx = offs_m // (OH * OW)

    n_mask = offs_n < OC

    K = IC * KH * KW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k  # [BLOCK_K]
        k_mask = k < K

        # Decompose k into (kh, kw, ic)
        ic = k % IC
        kw = (k // IC) % KW
        kh = k // (IC * KW)

        # Input gather: x[n, oh+kh, ow+kw, ic]
        ih = oh_idx[:, None] + kh[None, :]  # [BLOCK_M, BLOCK_K]
        iw = ow_idx[:, None] + kw[None, :]
        x_offset = (n_idx[:, None] * H * W * IC
                    + ih * W * IC
                    + iw * IC
                    + ic[None, :])
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        # Weight gather: w[oc, kh, kw, ic] -> [BLOCK_K, BLOCK_N]
        w_offset = (offs_n[None, :] * KH * KW * IC
                    + kh[:, None] * KW * IC
                    + kw[:, None] * IC
                    + ic[:, None])
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # Bias
    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b[None, :]

    # Divide
    acc = acc * INV_DIV

    # Leaky ReLU
    acc = tl.where(acc >= 0, acc, acc * NEG_SLOPE)

    # Store NHWC
    y_offset = (n_idx[:, None] * OH * OW * OC
                + oh_idx[:, None] * OW * OC
                + ow_idx[:, None] * OC
                + offs_n[None, :])
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + y_offset, acc, mask=y_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = float(divisor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        N, IC, H, W = x.shape
        KH = KW = self.kernel_size
        OC = self.out_channels
        OH = H - KH + 1
        OW = W - KW + 1

        # NHWC input
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        # Weight to (OC, KH, KW, IC)
        w = self.conv.weight.detach().to(x.device).contiguous()
        w_perm = w.permute(0, 2, 3, 1).contiguous()
        b = self.conv.bias.detach().to(x.device).contiguous()

        y_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        M = N * OH * OW
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OC, meta['BLOCK_N']))

        conv_div_lrelu_kernel[grid](
            x_nhwc, w_perm, b, y_nhwc,
            N, H, W, IC,
            OC, KH, KW,
            OH, OW,
            INV_DIV=1.0 / self.divisor,
            NEG_SLOPE=0.01,
        )

        y = y_nhwc.permute(0, 3, 1, 2).contiguous()
        return y