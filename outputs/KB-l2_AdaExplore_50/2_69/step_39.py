import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
    ],
    key=['N', 'OC', 'IC', 'H', 'W', 'KH', 'KW'],
)
@triton.jit
def conv_hs_relu_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    NHW = N * OH * OW
    mask_m = offs_m < NHW
    mask_n = offs_n < OC

    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    x_row_base = n_idx * stride_xn + oh_idx * stride_xh + ow_idx * stride_xw

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_idx = k_start + offs_k
        k_mask = k_idx < K_TOTAL

        ic_k = k_idx // (KH * KW)
        kr = k_idx % (KH * KW)
        kh_k = kr // KW
        kw_k = kr % KW

        x_k_off = ic_k * stride_xc + kh_k * stride_xh + kw_k * stride_xw
        x_off = x_row_base[:, None] + x_k_off[None, :]
        x_mask = mask_m[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        w_off = k_idx[:, None] + offs_n[None, :] * K_TOTAL
        w_mask = k_mask[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b_vals[None, :]

    out = tl.where(acc > 0.0, acc * tl.minimum(acc + 3.0, 6.0) * (1.0 / 6.0), 0.0)

    out_off = (n_idx * stride_on + oh_idx * stride_oh + ow_idx * stride_ow)[:, None] + offs_n[None, :] * stride_oc
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        N, IC, H, W = x.shape
        OC, _, KH, KW = weight.shape
        OH = H - KH + 1
        OW = W - KW + 1
        K_TOTAL = IC * KH * KW

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            triton.cdiv(N * OH * OW, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )

        conv_hs_relu_gemm_kernel[grid](
            x, weight, bias, out,
            N, IC, H, W,
            OC, OH, OW,
            KH, KW,
            K_TOTAL,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out