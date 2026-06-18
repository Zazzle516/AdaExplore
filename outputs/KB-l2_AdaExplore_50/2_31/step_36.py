import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Implicit im2col GEMM with NHWC layout, fused epilogue (min, bias, scale).
# A: input in NHWC, K dim = IC*KH*KW (loaded implicitly)
# B: weight transposed to (KH*KW*IC, OC)
# C: output in NHWC -> (N*OH*OW, OC)

AUTOTUNE_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=['M', 'N', 'K', 'OH', 'OW', 'IC', 'KH', 'KW'])
@triton.jit
def conv_fused_kernel(
    x_ptr, w_ptr, bconv_ptr, bias_ptr, out_ptr,
    M, N, K,
    N_BATCH, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    constant_value, scaling_factor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # M tile (output spatial * batch)
    pid_n = tl.program_id(1)  # N tile (output channels)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row index in M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channel
    offs_k = tl.arange(0, BLOCK_K)

    # Decode M index -> (n, oh, ow)
    OHW = OH * OW
    n_idx = offs_m // OHW
    rem = offs_m % OHW
    oh = rem // OW
    ow = rem % OW

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K = IC*KH*KW
    # Decode k -> (kh, kw, ic)
    # Weight layout: (KH*KW*IC, OC) so weight stride contiguous in OC
    # Input NHWC: x[n, ih, iw, ic], stride (IH*IW*IC, IW*IC, IC, 1)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        k_mask = k_idx < K

        # decode k -> kh, kw, ic
        kh = k_idx // (KW * IC)
        rem_k = k_idx % (KW * IC)
        kw = rem_k // IC
        ic = rem_k % IC

        # ih = oh + kh, iw = ow + kw  (no padding, stride=1, dilation=1)
        ih = oh[:, None] + kh[None, :]  # [BLOCK_M, BLOCK_K]
        iw = ow[:, None] + kw[None, :]

        # x address
        x_off = (n_idx[:, None] * (IH * IW * IC)
                 + ih * (IW * IC)
                 + iw * IC
                 + ic[None, :])
        x_valid = m_mask[:, None] & k_mask[None, :]
        a = tl.load(x_ptr + x_off, mask=x_valid, other=0.0)

        # weight address: w[k, oc]
        w_off = k_idx[:, None] * OC + offs_n[None, :]
        w_valid = k_mask[:, None] & n_mask[None, :]
        b = tl.load(w_ptr + w_off, mask=w_valid, other=0.0)

        acc += tl.dot(a, b, allow_tf32=False)

    # Add conv bias
    bc = tl.load(bconv_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bc[None, :]

    # min with constant
    acc = tl.minimum(acc, constant_value)

    # add extra bias (per output channel, shape (OC,1,1))
    bx = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bx[None, :]

    # scale
    acc = acc * scaling_factor

    # store: output NHWC (M, OC)
    out_off = offs_m[:, None] * OC + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv_fused(x_nhwc, w_kkic_oc, bconv, bias_extra, N, IC, IH, IW, OC, KH, KW, OH, OW,
               constant_value, scaling_factor):
    M = N * OH * OW
    Nn = OC
    K = IC * KH * KW
    out = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(Nn, meta['BLOCK_N']),
    )
    conv_fused_kernel[grid](
        x_nhwc, w_kkic_oc, bconv, bias_extra, out,
        M, Nn, K,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        float(constant_value), float(scaling_factor),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # x: (N, IC, H, W)
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        OC = self.out_channels
        OH = IH - KH + 1
        OW = IW - KW + 1

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        # Weight: (OC, IC, KH, KW) -> (KH, KW, IC, OC) -> (KH*KW*IC, OC)
        w = self.conv.weight  # (OC, IC, KH, KW)
        w_t = w.permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC)

        bconv = self.conv.bias.contiguous() if self.conv.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        bias_extra = self.bias.view(-1).contiguous()

        out_nhwc = conv_fused(
            x_nhwc, w_t, bconv, bias_extra,
            N, IC, IH, IW, OC, KH, KW, OH, OW,
            self.constant_value, self.scaling_factor,
        )
        # Convert back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out