import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ConvTranspose2d with stride=2, kernel=3, padding=1, output_padding=1
# Input:  [N, IC, H, W]
# Output: [N, OC, H*2, W*2]
# 
# For stride=2, k=3, padding=1, output_padding=1:
#   out_h = (in_h - 1)*2 - 2 + 3 + 1 = 2*in_h
#   For each output position (oh, ow), the contributing input positions are
#   those (ih, iw, kh, kw) such that ih*2 - 1 + kh = oh, iw*2 - 1 + kw = ow
#   -> kh = oh - 2*ih + 1, kw = ow - 2*iw + 1, with 0 <= kh,kw < 3
#   For each (oh, ow), there are at most ceil(3/2)=2 valid kh values and 2 valid kw values.
#
# We implement as a gather: for each output tile, loop over IC and accumulate.
# Use channels-last output layout for coalesced stores.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['N', 'IC', 'OC', 'H', 'W'],
)
@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,           # [N, IC, H, W]
    w_ptr,           # [IC, OC, KH, KW] (PyTorch ConvTranspose2d weight layout)
    bias_conv_ptr,   # [OC]
    bias_extra_ptr,  # [OC]
    out_ptr,         # [N, OH, OW, OC] (channels-last)
    inv_scale,
    N, IC, OC, H, W, OH, OW,
    BLOCK_M: tl.constexpr,  # spatial tile
    BLOCK_N: tl.constexpr,  # OC tile
    BLOCK_K: tl.constexpr,  # IC tile
):
    pid_m = tl.program_id(0)  # over N*OH*OW / BLOCK_M
    pid_n = tl.program_id(1)  # over OC / BLOCK_N

    M = N * OH * OW
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < OC

    # Decompose offs_m into (n, oh, ow)
    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    # For stride=2, k=3, pad=1: kh = oh - 2*ih + 1
    # ih = (oh + 1 - kh) / 2, requires (oh + 1 - kh) even and >= 0 and < H
    # The two valid kh values for a given oh: kh in {(oh+1) % 2, (oh+1) % 2 + 2}  -- if those are < 3
    # Specifically: if oh is even, oh+1 is odd, so kh must be odd: kh in {1}
    #               if oh is odd, oh+1 is even, so kh must be even: kh in {0, 2}
    # Wait: kh in [0, 3), and (oh+1-kh) must be even and nonneg and < 2H
    # Let's just iterate over all 9 (kh, kw) and mask.

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KH: tl.constexpr = 3
    KW: tl.constexpr = 3
    STRIDE: tl.constexpr = 2
    PAD: tl.constexpr = 1

    for kh in tl.static_range(KH):
        # ih * 2 = oh + 1 - kh  -> need (oh + 1 - kh) % 2 == 0 and >=0 and ih < H
        ih_num = oh_idx + PAD - kh
        ih = ih_num // STRIDE
        kh_valid = (ih_num % STRIDE == 0) & (ih_num >= 0) & (ih < H)
        for kw in tl.static_range(KW):
            iw_num = ow_idx + PAD - kw
            iw = iw_num // STRIDE
            kw_valid = (iw_num % STRIDE == 0) & (iw_num >= 0) & (iw < W)
            valid = kh_valid & kw_valid & m_mask  # [BLOCK_M]

            # For this (kh, kw) tap, we need to compute:
            # acc[m, n] += sum_ic x[n_idx, ic, ih, iw] * w[ic, oc, kh, kw]
            # This is a GEMM along IC.

            # Load x: [BLOCK_M, BLOCK_K] for each ic block
            # Load w: [BLOCK_K, BLOCK_N]

            # x offset: ((n_idx * IC + ic) * H + ih) * W + iw
            # base for each m: (n_idx * IC * H * W) + ih * W + iw
            x_base = n_idx * (IC * H * W) + ih * W + iw  # [BLOCK_M]

            # w offset: ((ic * OC) + oc) * KH * KW + kh * KW + kw
            w_kh_kw_off = kh * KW + kw

            for ic_start in range(0, IC, BLOCK_K):
                offs_k = ic_start + tl.arange(0, BLOCK_K)
                k_mask = offs_k < IC

                # x_ptrs: [BLOCK_M, BLOCK_K]
                x_ptrs = x_ptr + x_base[:, None] + offs_k[None, :] * (H * W)
                x_load_mask = valid[:, None] & k_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

                # w_ptrs: [BLOCK_K, BLOCK_N]
                # w[ic, oc, kh, kw] -> ic * (OC*KH*KW) + oc * (KH*KW) + w_kh_kw_off
                w_ptrs = w_ptr + offs_k[:, None] * (OC * KH * KW) + offs_n[None, :] * (KH * KW) + w_kh_kw_off
                w_load_mask = k_mask[:, None] & n_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Add biases
    bc = tl.load(bias_conv_ptr + offs_n, mask=n_mask, other=0.0)
    be = tl.load(bias_extra_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bc[None, :] + be[None, :]

    # clamp(0,1) -> *s -> clamp(0,1) -> /s
    # = clamp(0,1) of (clamp(0,1)*s) / s = min(1/s, clamp(0,1))
    # Just do it directly
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    # Since clamp produces values in [0,1], multiplying by s>=1 and clamping to 1 then div by s
    # gives min(acc, 1/s). For scaling_factor=2.0, that's min(acc, 0.5).
    # But we'll do generally:
    acc = acc  # already in [0,1]
    # multiply by scaling_factor implicitly: clamp(acc * s, 0, 1) / s
    # = min(acc, 1/s) since acc >= 0
    acc = tl.minimum(acc, inv_scale)

    # Store: out is channels-last [N, OH, OW, OC]
    # offset: m * OC + n  (where m = n_idx*OH*OW + oh*OW + ow)
    out_ptrs = out_ptr + offs_m[:, None] * OC + offs_n[None, :]
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


def conv_transpose_fused(x, weight, bias_conv, bias_extra, scaling_factor):
    N, IC, H, W = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w
    OH = H * 2
    OW = W * 2

    x = x.contiguous()
    weight = weight.contiguous()
    bias_conv = bias_conv.contiguous()
    bias_extra = bias_extra.contiguous().view(-1)

    # Output in channels-last [N, OH, OW, OC]
    out_cl = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

    M = N * OH * OW
    inv_scale = 1.0 / scaling_factor

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    conv_transpose_fused_kernel[grid](
        x, weight, bias_conv, bias_extra, out_cl,
        inv_scale,
        N, IC, OC, H, W, OH, OW,
    )

    # Convert back to [N, OC, OH, OW]
    out = out_cl.permute(0, 3, 1, 2).contiguous()
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        # Use custom kernel only for the supported config
        if (self.kernel_size == 3 and self.stride == 2 and self.padding == 1
                and self.output_padding == 1 and x.is_cuda and x.dtype == torch.float32):
            return conv_transpose_fused(
                x,
                self.conv_transpose.weight,
                self.conv_transpose.bias,
                self.bias,
                self.scaling_factor,
            )
        # Fallback
        x = self.conv_transpose(x)
        x = x + self.bias
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x * self.scaling_factor
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x / self.scaling_factor
        return x