import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_implicit_gemm_kernel(
    x_ptr,      # NHWC: (N, IH, IW, IC)
    w_ptr,      # (OC, KH, KW, IC) -- reshaped from (OC, IC, KH, KW) via permute
    b_ptr,      # (OC,)
    mult_ptr,   # (OC,)
    out_ptr,    # NHWC: (N, OH, OW, OC)
    N, IH, IW, IC,
    OH, OW,
    M, NN, K,   # M = OC, NN = N*OH*OW, K = IC*KH*KW
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC dim
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # N*OH*OW dim

    m_mask = offs_m < M
    n_mask = offs_n < NN

    # decompose offs_n into (n, oh, ow)
    OHW = OH * OW
    n_idx = offs_n // OHW
    rem = offs_n % OHW
    oh_idx = rem // OW
    ow_idx = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHW = KH * KW
    # K iterations: K = IC * KHW
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # decompose k into (ic, kh, kw)
        ic_k = offs_k // KHW
        rem_k = offs_k % KHW
        kh_k = rem_k // KW
        kw_k = rem_k % KW

        # Load x: for each (n, oh, ow) in offs_n, and each (ic, kh, kw) in offs_k
        # ih = oh + kh, iw = ow + kw
        ih = oh_idx[:, None] + kh_k[None, :]  # (BLOCK_N, BLOCK_K)
        iw = ow_idx[:, None] + kw_k[None, :]
        # x is NHWC: idx = n * IH*IW*IC + ih * IW*IC + iw * IC + ic
        x_idx = (n_idx[:, None] * (IH * IW * IC)
                 + ih * (IW * IC)
                 + iw * IC
                 + ic_k[None, :])
        x_load_mask = n_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_idx, mask=x_load_mask, other=0.0)  # (BLOCK_N, BLOCK_K)

        # Load w: (OC, KH, KW, IC) layout -> idx = oc * KHW*IC + kh*KW*IC + kw*IC + ic
        w_idx = (offs_m[:, None] * (KHW * IC)
                 + kh_k[None, :] * (KW * IC)
                 + kw_k[None, :] * IC
                 + ic_k[None, :])
        w_load_mask = m_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_idx, mask=w_load_mask, other=0.0)  # (BLOCK_M, BLOCK_K)

        # acc += w_tile @ x_tile.T  -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(w_tile, tl.trans(x_tile), allow_tf32=True)

    # Epilogue: bias + multiplier + leaky_relu + gelu
    bias = tl.load(b_ptr + offs_m, mask=m_mask, other=0.0)
    mult = tl.load(mult_ptr + offs_m, mask=m_mask, other=0.0)
    acc = acc + bias[:, None]
    acc = acc * mult[:, None]
    acc = tl.where(acc >= 0, acc, acc * 0.01)
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Store NHWC: out_idx = n * OH*OW*OC + oh*OW*OC + ow*OC + oc
    out_idx = (n_idx[None, :] * (OH * OW * M)
               + oh_idx[None, :] * (OW * M)
               + ow_idx[None, :] * M
               + offs_m[:, None])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to (OC, KH, KW, IC) for NHWC-friendly loads
        with torch.no_grad():
            w = self.conv.weight.detach()  # (OC, IC, KH, KW)
            w_perm = w.permute(0, 2, 3, 1).contiguous()  # (OC, KH, KW, IC)
        self.register_buffer('_w_perm', w_perm, persistent=False)
        self._weight_version = self.conv.weight._version

    def _maybe_update_weight(self):
        if self.conv.weight._version != self._weight_version:
            with torch.no_grad():
                w = self.conv.weight.detach()
                self._w_perm = w.permute(0, 2, 3, 1).contiguous()
            self._weight_version = self.conv.weight._version

    def forward(self, x):
        x = x.cuda()
        # Convert to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        self._maybe_update_weight()
        w_perm = self._w_perm
        b = self.conv.bias.contiguous()
        mult = self.multiplier.contiguous().view(-1)

        N, IH, IW, IC = x_nhwc.shape
        OC = w_perm.shape[0]
        KH = w_perm.shape[1]
        KW = w_perm.shape[2]
        OH = IH - KH + 1
        OW = IW - KW + 1

        M = OC
        NN = N * OH * OW
        K = IC * KH * KW

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(NN, meta['BLOCK_N']))

        conv_implicit_gemm_kernel[grid](
            x_nhwc, w_perm, b, mult, out_nhwc,
            N, IH, IW, IC,
            OH, OW,
            M, NN, K,
            KH=KH, KW=KW,
        )

        # Permute back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out