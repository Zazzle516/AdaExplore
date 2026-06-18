import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mish(x):
    # mish(x) = x * tanh(softplus(x))
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    e = tl.exp(2.0 * sp)
    t = (e - 1.0) / (e + 1.0)
    return x * t


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['N', 'OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_nhwc_mish2_kernel(
    x_ptr,   # [N, IH, IW, IC]
    w_ptr,   # [OC, KH, KW, IC]
    b_ptr,   # [OC]
    y_ptr,   # [N, OH, OW, OC]
    N, IC, IH, IW,
    OC, KH, KW,
    OH, OW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # over (N * OH * OW) / BLOCK_M
    pid_n = tl.program_id(1)  # over OC / BLOCK_N

    M = N * OH * OW
    K = IC * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # decode m to (n, oh, ow)
    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    m_mask = offs_m < M
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # decode k -> (ic, kh, kw)
        ic_idx = offs_k // (KH * KW)
        krem = offs_k % (KH * KW)
        kh_idx = krem // KW
        kw_idx = krem % KW

        # input gather: ih = oh + kh, iw = ow + kw  (no padding)
        ih = oh_idx[:, None] + kh_idx[None, :]  # [M, K]
        iw = ow_idx[:, None] + kw_idx[None, :]

        # x is [N, IH, IW, IC] contiguous
        x_offs = (n_idx[:, None] * IH * IW * IC
                  + ih * IW * IC
                  + iw * IC
                  + ic_idx[None, :])
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # weight is [OC, KH, KW, IC] contiguous; we need w[oc, kh, kw, ic] for each (k, n)
        # offs for [BLOCK_K, BLOCK_N]
        w_offs = (offs_n[None, :] * KH * KW * IC
                  + kh_idx[:, None] * KW * IC
                  + kw_idx[:, None] * IC
                  + ic_idx[:, None])
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(x_tile, w_tile)

    # bias
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)  # [BLOCK_N]
    acc = acc + bias[None, :]

    # double mish
    acc = _mish(acc)
    acc = _mish(acc)

    # store: y is [N, OH, OW, OC] contiguous
    y_offs = (n_idx[:, None] * OH * OW * OC
              + oh_idx[:, None] * OW * OC
              + ow_idx[:, None] * OC
              + offs_n[None, :])
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + y_offs, acc, mask=y_mask)


def conv2d_mish_mish_nhwc(x_nhwc, w_nhwc, b, OH, OW):
    N, IH, IW, IC = x_nhwc.shape
    OC, KH, KW, _ = w_nhwc.shape
    y = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)

    M = N * OH * OW
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )
    conv2d_nhwc_mish2_kernel[grid](
        x_nhwc, w_nhwc, b, y,
        N, IC, IH, IW,
        OC, KH, KW,
        OH, OW,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # cache NHWC weight
        self._weight_nhwc = None

    def _get_weight_nhwc(self):
        w = self.conv.weight  # [OC, IC, KH, KW]
        w_nhwc = w.permute(0, 2, 3, 1).contiguous()
        return w_nhwc

    def forward(self, x):
        # x: [N, IC, IH, IW]
        if x.device.type != 'cuda':
            x = x.cuda()
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        OC = self.out_channels

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w_nhwc = self._get_weight_nhwc().to(x.device)
        b = self.conv.bias.to(x.device).contiguous()

        y_nhwc = conv2d_mish_mish_nhwc(x_nhwc, w_nhwc, b, OH, OW)
        y = y_nhwc.permute(0, 3, 1, 2).contiguous()
        return y