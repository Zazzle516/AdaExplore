import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OHW', 'K'],
)
@triton.jit
def _conv_fused_kernel(
    x_ptr, w_ptr, cbias_ptr, ebias_ptr, out_ptr,
    N, IC, H, W, OC, OH, OW, KH, KW,
    OHW, K,
    constant_value, scaling_factor,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program over (N, OC_tile, OHW_tile)
    pid_n_batch = tl.program_id(0)  # batch index
    pid_oc = tl.program_id(1)       # OC tile
    pid_ohw = tl.program_id(2)      # OHW tile

    # rows = OC dim (M), cols = OHW dim (N)
    offs_m = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_n = pid_ohw * BLOCK_N + tl.arange(0, BLOCK_N)  # OHW indices

    mask_m = offs_m < OC
    mask_n = offs_n < OHW

    # Decode output spatial
    oh = offs_n // OW
    ow = offs_n % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop: IC * KH * KW
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Decode k -> (ic, kh, kw)
        ic = offs_k // (KH * KW)
        rem = offs_k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # Weight: (OC, K) layout -> w[m, k]
        w_ptrs = w_ptr + offs_m[:, None] * K + offs_k[None, :]
        w_mask = mask_m[:, None] & mask_k[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Input gather: x[n_batch, ic, oh+kh, ow+kw]
        ih = oh[None, :] + kh[:, None]  # (BLOCK_K, BLOCK_N)
        iw = ow[None, :] + kw[:, None]  # (BLOCK_K, BLOCK_N)
        ic_b = ic[:, None]              # (BLOCK_K, 1)

        x_ptrs = (x_ptr
                  + pid_n_batch * stride_xn
                  + ic_b * stride_xc
                  + ih * stride_xh
                  + iw * stride_xw)

        x_mask = (mask_k[:, None]) & (mask_n[None, :])
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(w, x, allow_tf32=False)

    # Epilogue: add conv bias, min, add ebias, scale
    cbias = tl.load(cbias_ptr + offs_m, mask=mask_m, other=0.0)
    ebias = tl.load(ebias_ptr + offs_m, mask=mask_m, other=0.0)

    out = acc + cbias[:, None]
    out = tl.minimum(out, constant_value)
    out = (out + ebias[:, None]) * scaling_factor

    # Store: out[n_batch, oc, oh, ow]
    out_ptrs = (out_ptr
                + pid_n_batch * stride_on
                + offs_m[:, None] * stride_oc
                + oh[None, :] * stride_oh
                + ow[None, :] * stride_ow)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, out, mask=out_mask)


def conv2d_fused(x, weight_2d, conv_bias, ebias, constant_value, scaling_factor,
                 KH, KW):
    N, IC, H, W = x.shape
    OC = weight_2d.shape[0]
    K = weight_2d.shape[1]
    OH = H - KH + 1
    OW = W - KW + 1
    OHW = OH * OW

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        N,
        triton.cdiv(OC, meta['BLOCK_M']),
        triton.cdiv(OHW, meta['BLOCK_N']),
    )

    _conv_fused_kernel[grid](
        x, weight_2d, conv_bias, ebias, out,
        N, IC, H, W, OC, OH, OW, KH, KW,
        OHW, K,
        float(constant_value), float(scaling_factor),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
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
        x = x.contiguous()
        KH = self.kernel_size
        KW = self.kernel_size
        OC = self.out_channels
        # Reshape conv weight to (OC, IC*KH*KW)
        weight_2d = self.conv.weight.view(OC, -1).contiguous()
        conv_bias = self.conv.bias.contiguous()
        ebias = self.bias.view(-1).contiguous()
        return conv2d_fused(
            x, weight_2d, conv_bias, ebias,
            self.constant_value, self.scaling_factor,
            KH, KW,
        )