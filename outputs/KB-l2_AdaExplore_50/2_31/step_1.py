import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OUT_SPATIAL', 'IC_KH_KW'],
)
@triton.jit
def conv2d_fused_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    constant_value, scaling_factor,
    OUT_SPATIAL, IC_KH_KW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program_id(0): batch
    # program_id(1): OC tile (M)
    # program_id(2): output spatial tile (N)
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)  # output spatial

    oh = offs_s // OW
    ow = offs_s % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K = IC_KH_KW
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        ic = offs_k // (KH * KW)
        kh = (offs_k % (KH * KW)) // KW
        kw = offs_k % KW

        # weight: [OC, IC, KH, KW]
        w_ptrs = w_ptr + (offs_m[:, None] * stride_wo
                          + ic[None, :] * stride_wi
                          + kh[None, :] * stride_wh
                          + kw[None, :] * stride_ww)
        w_mask = (offs_m[:, None] < OC) & k_mask[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # input gather: x[n, ic, oh+kh, ow+kw]
        ih = oh[None, :] + kh[:, None]
        iw = ow[None, :] + kw[:, None]
        x_ptrs = x_ptr + (pid_n * stride_xn
                          + ic[:, None] * stride_xc
                          + ih * stride_xh
                          + iw * stride_xw)
        x_mask = k_mask[:, None] & (offs_s[None, :] < OUT_SPATIAL) & (ih < H) & (iw < W)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(w_tile, x_tile)

    # bias from conv
    b_vals = tl.load(b_ptr + offs_m, mask=offs_m < OC, other=0.0)
    acc += b_vals[:, None]

    # min with constant
    acc = tl.minimum(acc, constant_value)

    # additional bias [OC, 1, 1]
    extra_bias = tl.load(bias_ptr + offs_m, mask=offs_m < OC, other=0.0)
    acc += extra_bias[:, None]

    # scale
    acc = acc * scaling_factor

    # store
    out_ptrs = out_ptr + (pid_n * stride_on
                          + offs_m[:, None] * stride_oc
                          + oh[None, :] * stride_oh
                          + ow[None, :] * stride_ow)
    out_mask = (offs_m[:, None] < OC) & (offs_s[None, :] < OUT_SPATIAL)
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        bias_extra = self.bias.contiguous().view(-1)

        N, IC, H, W = x.shape
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        OUT_SPATIAL = OH * OW
        IC_KH_KW = IC * KH * KW

        grid = lambda meta: (
            N,
            (OC + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
            (OUT_SPATIAL + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
        )

        conv2d_fused_kernel[grid](
            x, w, b, bias_extra, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            float(self.constant_value), float(self.scaling_factor),
            OUT_SPATIAL, IC_KH_KW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out