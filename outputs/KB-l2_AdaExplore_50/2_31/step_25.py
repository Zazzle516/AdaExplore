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
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'K'],
)
@triton.jit
def conv2d_fused_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    KH, KW,
    OUT_HW, K,
    constant_value, scaling_factor,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # pid_m -> OC tile, pid_n -> spatial tile, pid_batch -> N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial indices (oh*OW + ow)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < OC
    mask_n = offs_n < OUT_HW

    oh = offs_n // OW
    ow = offs_n % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K
        # decompose k_idx -> (ic, kh, kw)
        kw = k_idx % KW
        khc = k_idx // KW
        kh = khc % KH
        ic = khc // KH

        # Load weight: w[oc, ic, kh, kw], shape (BLOCK_M, BLOCK_K)
        w_offset = (offs_m[:, None] * stride_wo
                    + ic[None, :] * stride_wi
                    + kh[None, :] * stride_wh
                    + kw[None, :] * stride_ww)
        w_mask = mask_m[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        # Load input: x[n, ic, oh+kh, ow+kw], shape (BLOCK_K, BLOCK_N)
        ih = oh[None, :] + kh[:, None]
        iw = ow[None, :] + kw[:, None]
        x_offset = (pid_b * stride_xn
                    + ic[:, None] * stride_xc
                    + ih * stride_xh
                    + iw * stride_xw)
        x_mask = (k_mask[:, None]
                  & mask_n[None, :]
                  & (ih < H) & (iw < W))
        x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

        acc += tl.dot(w_vals, x_vals, allow_tf32=False, out_dtype=tl.float32)

    # add conv bias (per-OC)
    b_vals = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + b_vals[:, None]

    # min with constant
    acc = tl.minimum(acc, constant_value)

    # add extra bias (per-OC since bias_shape is (OC,1,1))
    bias_vals = tl.load(bias_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + bias_vals[:, None]

    # scale
    acc = acc * scaling_factor

    # store
    out_offset = (pid_b * stride_on
                  + offs_m[:, None] * stride_oc
                  + oh[None, :] * stride_oh
                  + ow[None, :] * stride_ow)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


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
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        bias = self.bias.contiguous().view(-1)

        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        OUT_HW = OH * OW
        K = IC * KH * KW

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda META: (
            triton.cdiv(OC, META['BLOCK_M']),
            triton.cdiv(OUT_HW, META['BLOCK_N']),
            N,
        )

        conv2d_fused_kernel[grid](
            x, w, b, bias, out,
            N, IC, H, W,
            OC, OH, OW,
            KH, KW,
            OUT_HW, K,
            float(self.constant_value), float(self.scaling_factor),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out