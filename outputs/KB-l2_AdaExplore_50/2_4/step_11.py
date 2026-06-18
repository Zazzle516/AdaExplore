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
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['N', 'OC', 'OH', 'OW', 'IC'],
)
@triton.jit
def conv2d_nhwc_mish2_kernel(
    x_ptr,   # [N, IH, IW, IC]
    w_ptr,   # [OC, 3, 3, IC]
    b_ptr,   # [OC]
    y_ptr,   # [N, OH, OW, OC]
    N, IC, IH, IW,
    OC,
    OH, OW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = N * OH * OW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # decode m to (n, oh, ow)
    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    m_mask = offs_m < M

    # Hoisted base offsets
    # x base = n*IH*IW*IC + oh*IW*IC + ow*IC  (kh,kw added per tap)
    x_base = (n_idx * IH * IW * IC + oh_idx * IW * IC + ow_idx * IC)  # [BLOCK_M]
    # w base for oc range
    w_base_oc = offs_n * (9 * IC)  # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    IWIC = IW * IC

    # Loop over IC in tiles of BLOCK_K, fully unroll 9 taps
    for ic_start in range(0, IC, BLOCK_K):
        offs_k = ic_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < IC

        # 9 taps unrolled: (kh, kw) in [0..2] x [0..2]
        # tap (0,0)
        x_off = x_base[:, None] + 0 * IWIC + 0 * IC + offs_k[None, :]
        w_off = w_base_oc[None, :] + (0 * 3 + 0) * IC + offs_k[:, None]
        x_tile = tl.load(x_ptr + x_off, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_ptr + w_off, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(x_tile, w_tile)

        # tap (0,1)
        x_off = x_base[:, None] + 0 * IWIC + 1 * IC + offs_k[None, :]
        w_off = w_base_oc[None, :] + (0 * 3 + 1) * IC + offs_k[:, None]
        x_tile = tl.load(x_ptr + x_off, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_ptr + w_off, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(x_tile, w_tile)

        # tap (0,2)
        x_off = x_base[:, None] + 0 * IWIC + 2 * IC + offs_k[None, :]
        w_off = w_base_oc[None, :] + (0 * 3 + 2) * IC + offs_k[:, None]
        x_tile = tl.load(x_ptr + x_off, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_ptr + w_off, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(x_tile, w_tile)

        # tap (1,0)
        x_off = x_base[:, None] + 1 * IWIC + 0 * IC + offs_k[None, :]
        w_off = w_base_oc[None, :] + (1 * 3 + 0) * IC + offs_k[:, None]
        x_tile = tl.load(x_ptr + x_off, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_ptr + w_off, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(x_tile, w_tile)

        # tap (1,1)
        x_off = x_base[:, None] + 1 * IWIC + 1 * IC + offs_k[None, :]
        w_off = w_base_oc[None, :] + (1 * 3 + 1) * IC + offs_k[:, None]
        x_tile = tl.load(x_ptr + x_off, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_ptr + w_off, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(x_tile, w_tile)

        # tap (1,2)
        x_off = x_base[:, None] + 1 * IWIC + 2 * IC + offs_k[None, :]
        w_off = w_base_oc[None, :] + (1 * 3 + 2) * IC + offs_k[:, None]
        x_tile = tl.load(x_ptr + x_off, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_ptr + w_off, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(x_tile, w_tile)

        # tap (2,0)
        x_off = x_base[:, None] + 2 * IWIC + 0 * IC + offs_k[None, :]
        w_off = w_base_oc[None, :] + (2 * 3 + 0) * IC + offs_k[:, None]
        x_tile = tl.load(x_ptr + x_off, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_ptr + w_off, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(x_tile, w_tile)

        # tap (2,1)
        x_off = x_base[:, None] + 2 * IWIC + 1 * IC + offs_k[None, :]
        w_off = w_base_oc[None, :] + (2 * 3 + 1) * IC + offs_k[:, None]
        x_tile = tl.load(x_ptr + x_off, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_ptr + w_off, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(x_tile, w_tile)

        # tap (2,2)
        x_off = x_base[:, None] + 2 * IWIC + 2 * IC + offs_k[None, :]
        w_off = w_base_oc[None, :] + (2 * 3 + 2) * IC + offs_k[:, None]
        x_tile = tl.load(x_ptr + x_off, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_tile = tl.load(w_ptr + w_off, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(x_tile, w_tile)

    # bias
    bias = tl.load(b_ptr + offs_n)  # [BLOCK_N]
    acc = acc + bias[None, :]

    # double mish
    acc = _mish(acc)
    acc = _mish(acc)

    # store: y is [N, OH, OW, OC] contiguous
    y_offs = (n_idx[:, None] * OH * OW * OC
              + oh_idx[:, None] * OW * OC
              + ow_idx[:, None] * OC
              + offs_n[None, :])
    tl.store(y_ptr + y_offs, acc, mask=m_mask[:, None])


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

        y_nhwc = conv2d_mish_mish_nhwc(x_nhwc, w_nhwc, b, OH, OW)
        y = y_nhwc.permute(0, 3, 1, 2).contiguous()
        return y