import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


CONV_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=CONV_CONFIGS, key=['N', 'OC', 'IC', 'H', 'W', 'KH', 'KW'])
@triton.jit
def _conv2d_mish_nhwc_kernel(
    x_ptr,         # [N, H, W, IC] (NHWC)
    w_ptr,         # [OC, KH, KW, IC] (OC, KH, KW, IC) flattened
    b_ptr,         # [OC]
    out_ptr,       # [N, OH, OW, OC] (NHWC)
    N, IC, H, W, OC, KH, KW, OH, OW,
    BLOCK_M: tl.constexpr,   # tile over (N*OH*OW)
    BLOCK_N: tl.constexpr,   # tile over OC
    BLOCK_K: tl.constexpr,   # tile over IC (contiguous)
):
    pid_m = tl.program_id(0)  # (N*OH*OW) tile
    pid_n = tl.program_id(1)  # OC tile

    M = N * OH * OW
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial+batch
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC

    # decode m -> n, oh, ow
    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    m_mask = offs_m < M
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # Loop over (kh, kw, ic-tile)
    for kh in range(0, KH):
        for kw in range(0, KW):
            h_in = oh_idx + kh    # [BLOCK_M]
            w_in = ow_idx + kw    # [BLOCK_M]
            # base offset for each row: n * H*W*IC + h_in * W*IC + w_in * IC
            x_row_base = (n_idx * (H * W * IC)
                          + h_in * (W * IC)
                          + w_in * IC)  # [BLOCK_M]
            # weight base for (kh,kw): w_ptr is [OC, KH, KW, IC]
            w_col_base = (offs_n * (KH * KW * IC)
                          + kh * (KW * IC)
                          + kw * IC)  # [BLOCK_N]

            for ic_start in range(0, IC, BLOCK_K):
                k_idx = ic_start + offs_k  # [BLOCK_K]
                k_mask = k_idx < IC

                # Load x: shape [BLOCK_M, BLOCK_K]
                x_off = x_row_base[:, None] + k_idx[None, :]
                x_mask = m_mask[:, None] & k_mask[None, :]
                x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                # Load w: shape [BLOCK_K, BLOCK_N]   (transposed: ic varies fast in K)
                w_off = k_idx[:, None] + w_col_base[None, :]
                w_mask_ = k_mask[:, None] & n_mask[None, :]
                w_vals = tl.load(w_ptr + w_off, mask=w_mask_, other=0.0)

                acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    # add bias
    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b[None, :]

    # Mish fused: x * tanh(softplus(x))
    sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    e2 = tl.exp(-2.0 * sp)
    t = (1.0 - e2) / (1.0 + e2)
    out = acc * t

    # Store to NHWC output [N, OH, OW, OC]
    # offset = m * OC + offs_n
    out_off = offs_m[:, None] * OC + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=out_mask)


def conv2d_mish_nhwc(x_nchw, weight, bias):
    """
    x_nchw: [N, IC, H, W]
    weight: [OC, IC, KH, KW]
    bias: [OC]
    Returns: NHWC tensor [N, OH, OW, OC] (after conv+mish)
    """
    N, IC, H, W = x_nchw.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    # Convert input to NHWC contiguous
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()  # [N,H,W,IC]
    # Convert weight to [OC, KH, KW, IC] contiguous
    w_oc_khkw_ic = weight.permute(0, 2, 3, 1).contiguous()  # [OC,KH,KW,IC]

    out_nhwc = torch.empty((N, OH, OW, OC), device=x_nchw.device, dtype=x_nchw.dtype)

    grid = lambda meta: (
        triton.cdiv(N * OH * OW, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    _conv2d_mish_nhwc_kernel[grid](
        x_nhwc, w_oc_khkw_ic, bias, out_nhwc,
        N, IC, H, W, OC, KH, KW, OH, OW,
    )

    return out_nhwc  # [N, OH, OW, OC]


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = x.contiguous()
        if x.is_cuda:
            # Conv + Mish in NHWC layout
            y_nhwc = conv2d_mish_nhwc(x, self.conv.weight, self.conv.bias)
            # Convert to NCHW for BN
            y = y_nhwc.permute(0, 3, 1, 2).contiguous()
        else:
            y = self.conv(x)
            y = y * torch.tanh(F.softplus(y))
        y = self.bn(y)
        return y