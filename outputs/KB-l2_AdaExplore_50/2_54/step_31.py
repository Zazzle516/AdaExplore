import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['OH', 'OW', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_nhwc_kloop_kernel(
    x_ptr,        # input NHWC: (N, IH, IW, IC)
    w_ptr,        # weight (OC, KH, KW, IC) contiguous, premultiplied by mult
    b_ptr,        # bias (OC,) premultiplied by mult
    out_ptr,      # output NHWC: (N, OH, OW, OC)
    N, IH, IW, IC,
    OH, OW, OC,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)             # batch
    pid_m = tl.program_id(1)             # output spatial tile
    pid_oc = tl.program_id(2)            # output channel tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < (OH * OW)
    n_mask = offs_n < OC

    oh = offs_m // OW
    ow = offs_m % OW

    x_batch_base = pid_n * IH * IW * IC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # weight layout: (OC, KH, KW, IC) -> stride for OC = KH*KW*IC
    W_OC_STRIDE = KH * KW * IC

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_idx = oh + kh    # (M,)
            iw_idx = ow + kw    # (M,)
            x_row_base = x_batch_base + (ih_idx * IW + iw_idx) * IC  # (M,)
            w_kh_kw_base = (kh * KW + kw) * IC  # scalar offset within OC

            for k0 in range(0, IC, BLOCK_K):
                k_idx = k0 + offs_k                       # (K,)

                x_offsets = x_row_base[:, None] + k_idx[None, :]
                x_block = tl.load(x_ptr + x_offsets, mask=m_mask[:, None], other=0.0)

                w_offsets = offs_n[None, :] * W_OC_STRIDE + w_kh_kw_base + k_idx[:, None]
                w_block = tl.load(w_ptr + w_offsets, mask=n_mask[None, :], other=0.0)

                acc += tl.dot(x_block, w_block, allow_tf32=True)

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # LeakyReLU(0.01)
    acc = tl.where(acc >= 0, acc, acc * 0.01)

    # GELU exact
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # store NCHW: (N, OC, OH, OW); pid_n batch, offs_n channel, offs_m spatial
    out_offsets = (pid_n * OC * OH * OW
                   + offs_n[None, :] * (OH * OW)
                   + offs_m[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, tuple):
            self.KH, self.KW = kernel_size
        else:
            self.KH = self.KW = kernel_size

        self._cached = False

    def _build_cache(self, device, dtype):
        with torch.no_grad():
            mult = self.multiplier.detach().to(device=device, dtype=dtype).view(-1)  # (OC,)
            w = self.conv.weight.detach().to(device=device, dtype=dtype)  # (OC, IC, KH, KW)
            b = self.conv.bias.detach().to(device=device, dtype=dtype)    # (OC,)

            w_scaled = w * mult.view(-1, 1, 1, 1)
            b_scaled = b * mult

            # Reorder weight (OC, IC, KH, KW) -> (OC, KH, KW, IC), contiguous
            w_re = w_scaled.permute(0, 2, 3, 1).contiguous()

        self._w = w_re
        self._b = b_scaled.contiguous()
        self._cached = True

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda(non_blocking=True)
        dtype = x.dtype
        device = x.device

        if (not self._cached) or (self._w.device != device) or (self._w.dtype != dtype):
            self._build_cache(device, dtype)

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.KH
        KW = self.KW
        OH = IH - KH + 1
        OW = IW - KW + 1

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        out = torch.empty((N, OC, OH, OW), device=device, dtype=dtype)

        grid = lambda META: (
            N,
            triton.cdiv(OH * OW, META['BLOCK_M']),
            triton.cdiv(OC, META['BLOCK_N']),
        )

        conv_nhwc_kloop_kernel[grid](
            x_nhwc, self._w, self._b, out,
            N, IH, IW, IC,
            OH, OW, OC,
            KH=KH, KW=KW,
        )

        return out