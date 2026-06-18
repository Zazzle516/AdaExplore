import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['OH', 'OW', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_nhwc_gemm_kernel(
    x_ptr,        # input NHWC: (N, IH, IW, IC)
    w_ptr,        # weight reshaped (OC, IC*KH*KW), already premultiplied by mult
    b_ptr,        # bias (OC,) already premultiplied by mult
    out_ptr,      # output NCHW: (N, OC, OH, OW)
    N, IH, IW, IC,
    OH, OW, OC,
    KH: tl.constexpr, KW: tl.constexpr,
    K_TOTAL: tl.constexpr,    # IC * KH * KW
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)             # batch
    pid_m = tl.program_id(1)             # output spatial tile
    pid_oc = tl.program_id(2)            # output channel tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # spatial idx
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # output channel idx
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < (OH * OW)
    n_mask = offs_n < OC

    oh = offs_m // OW
    ow = offs_m % OW

    # base pointer for this batch in NHWC layout
    x_batch_base = pid_n * IH * IW * IC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHW = KH * KW

    for k0 in range(0, K_TOTAL, BLOCK_K):
        k_idx = k0 + offs_k                                # (BLOCK_K,)
        k_valid = k_idx < K_TOTAL

        # decompose k -> (ic, kh, kw)
        ic = k_idx // KHW
        rem = k_idx % KHW
        kh = rem // KW
        kw_ = rem % KW

        # input row positions per (m,k)
        ih_idx = oh[:, None] + kh[None, :]   # (M, K)
        iw_idx = ow[:, None] + kw_[None, :]  # (M, K)

        x_offsets = x_batch_base + (ih_idx * IW + iw_idx) * IC + ic[None, :]
        x_mask = m_mask[:, None] & k_valid[None, :]
        x_block = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)  # (M, K)

        # weight: (OC, K_TOTAL) -> load (K, N)
        w_offsets = offs_n[None, :] * K_TOTAL + k_idx[:, None]
        w_mask = n_mask[None, :] & k_valid[:, None]
        w_block = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)  # (K, N)

        acc += tl.dot(x_block, w_block, allow_tf32=True)

    # epilogue: bias (already includes multiplier)
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # LeakyReLU
    acc = tl.where(acc >= 0, acc, acc * 0.01)

    # GELU exact
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # store NCHW
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
        # Fold multiplier into weight + bias.
        # weight: (OC, IC, KH, KW), multiplier: (OC, 1, 1)
        with torch.no_grad():
            mult = self.multiplier.detach().to(device=device, dtype=dtype).view(-1)  # (OC,)
            w = self.conv.weight.detach().to(device=device, dtype=dtype)  # (OC, IC, KH, KW)
            b = self.conv.bias.detach().to(device=device, dtype=dtype)    # (OC,)

            w_scaled = w * mult.view(-1, 1, 1, 1)        # (OC, IC, KH, KW)
            b_scaled = b * mult                          # (OC,)

            # Reorder weight to (OC, KH, KW, IC) then flatten last 3 to K = IC*KH*KW
            # so that K decomposes as ic varying fastest (matches our k = ic + kw_*IC ... )
            # Wait — in kernel we decompose: ic = k // (KH*KW); kh = (k%KHW)//KW; kw = k%KW
            # So the layout along K is [ic][kh][kw], i.e. weight (OC, IC, KH, KW) flattened directly.
            OC = w_scaled.shape[0]
            w_flat = w_scaled.reshape(OC, -1).contiguous()  # (OC, IC*KH*KW)

        self._w_flat = w_flat
        self._b_scaled = b_scaled.contiguous()
        self._cached = True

    def forward(self, x):
        x = x.cuda(non_blocking=True) if not x.is_cuda else x
        dtype = x.dtype
        device = x.device

        if (not self._cached) or (self._w_flat.device != device) or (self._w_flat.dtype != dtype):
            self._build_cache(device, dtype)

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.KH
        KW = self.KW
        OH = IH - KH + 1
        OW = IW - KW + 1

        # NHWC contiguous input
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        out = torch.empty((N, OC, OH, OW), device=device, dtype=dtype)

        K_TOTAL = IC * KH * KW

        grid = lambda META: (
            N,
            triton.cdiv(OH * OW, META['BLOCK_M']),
            triton.cdiv(OC, META['BLOCK_N']),
        )

        conv_nhwc_gemm_kernel[grid](
            x_nhwc, self._w_flat, self._b_scaled, out,
            N, IH, IW, IC,
            OH, OW, OC,
            KH=KH, KW=KW,
            K_TOTAL=K_TOTAL,
        )
        return out