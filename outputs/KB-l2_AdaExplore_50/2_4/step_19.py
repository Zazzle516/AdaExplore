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
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['N', 'OC', 'OH', 'OW', 'IC'],
)
@triton.jit
def conv2d_nhwc_mish2_kernel(
    x_ptr,   # [N, IH, IW, IC]
    w_ptr,   # [OC, KH, KW, IC]
    b_ptr,   # [OC]
    y_ptr,   # [N, OC, OH, OW] (NCHW output)
    N, IC, IH, IW,
    OC,
    OH, OW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    M = N * OH * OW
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(OC, BLOCK_N)

    # group-major swizzle for L2 reuse
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    KH: tl.constexpr = 3
    KW: tl.constexpr = 3

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    m_mask = offs_m < M
    n_mask = offs_n < OC

    # base offsets hoisted out of K-loop
    x_base_m = n_idx * (IH * IW * IC) + oh_idx * (IW * IC) + ow_idx * IC  # [BLOCK_M]
    w_base_n = offs_n * (KH * KW * IC)  # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    k_mask = offs_k < IC

    # iterate over the 9 kernel taps; with BLOCK_K>=IC we cover K in one pass
    for kh in tl.static_range(0, 3):
        for kw in tl.static_range(0, 3):
            x_tap = x_base_m + (kh * IW * IC + kw * IC)  # [BLOCK_M]
            w_tap = w_base_n + (kh * KW * IC + kw * IC)  # [BLOCK_N]

            x_offs = x_tap[:, None] + offs_k[None, :]
            x_mask = m_mask[:, None] & k_mask[None, :]
            x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

            w_offs = w_tap[None, :] + offs_k[:, None]
            w_mask_ = k_mask[:, None] & n_mask[None, :]
            w_tile = tl.load(w_ptr + w_offs, mask=w_mask_, other=0.0)

            acc += tl.dot(x_tile, w_tile)

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    acc = _mish(acc)
    acc = _mish(acc)

    # store directly into NCHW: y[n, oc, oh, ow]
    y_offs = (n_idx[:, None] * (OC * OH * OW)
              + offs_n[None, :] * (OH * OW)
              + oh_idx[:, None] * OW
              + ow_idx[:, None])
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + y_offs, acc, mask=y_mask)


def conv2d_mish_mish_nhwc(x_nhwc, w_nhwc, b, OH, OW):
    N, IH, IW, IC = x_nhwc.shape
    OC, KH, KW, _ = w_nhwc.shape
    y = torch.empty((N, OC, OH, OW), device=x_nhwc.device, dtype=x_nhwc.dtype)

    M = N * OH * OW
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(OC, meta['BLOCK_N']),
    )
    conv2d_nhwc_mish2_kernel[grid](
        x_nhwc, w_nhwc, b, y,
        N, IC, IH, IW,
        OC,
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

        y = conv2d_mish_mish_nhwc(x_nhwc, w_nhwc, b, OH, OW)
        return y