import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_fused_kernel(
    x_ptr, w_ptr, b_conv_ptr, b_extra_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    constant_value, scaling_factor,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,  # output spatial tile (per N)
    BLOCK_N: tl.constexpr,  # OC tile
    BLOCK_K: tl.constexpr,  # IC tile
):
    pid_m = tl.program_id(0)  # over N * (OH*OW / BLOCK_M)
    pid_n = tl.program_id(1)  # over OC tile

    num_pid_spatial = tl.cdiv(OH * OW, BLOCK_M)
    n_idx = pid_m // num_pid_spatial
    sp_idx = pid_m % num_pid_spatial

    offs_m = sp_idx * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial pos in OH*OW
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # oc
    offs_k = tl.arange(0, BLOCK_K)                      # ic

    oh = offs_m // OW
    ow = offs_m % OW

    mask_m = offs_m < (OH * OW)
    mask_n = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kh, kw, ic
    for kh in range(0, KH):
        ih = oh + kh  # padding=0
        for kw in range(0, KW):
            iw = ow + kw
            for ic_start in range(0, IC, BLOCK_K):
                ic_idx = ic_start + offs_k
                mask_k = ic_idx < IC

                # x: [BLOCK_M, BLOCK_K]
                x_offset = (n_idx * stride_xn
                            + ic_idx[None, :] * stride_xc
                            + ih[:, None] * stride_xh
                            + iw[:, None] * stride_xw)
                x_mask = mask_m[:, None] & mask_k[None, :]
                x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

                # w: [BLOCK_K, BLOCK_N]
                w_offset = (offs_n[None, :] * stride_wo
                            + ic_idx[:, None] * stride_wi
                            + kh * stride_wkh
                            + kw * stride_wkw)
                w_mask = mask_k[:, None] & mask_n[None, :]
                w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Add conv bias
    bconv = tl.load(b_conv_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bconv[None, :]

    # min with constant
    acc = tl.minimum(acc, constant_value)

    # Add extra bias (shape OC,1,1 -> per OC)
    bextra = tl.load(b_extra_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bextra[None, :]

    # scale
    acc = acc * scaling_factor

    # Store
    out_offset = (n_idx * stride_on
                  + offs_n[None, :] * stride_oc
                  + oh[:, None] * stride_oh
                  + ow[:, None] * stride_ow)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = float(constant_value)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = float(scaling_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b_conv = self.conv.bias.contiguous().cuda()
        b_extra = self.bias.view(-1).contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        def grid(meta):
            num_sp = triton.cdiv(OH * OW, meta['BLOCK_M'])
            return (N * num_sp, triton.cdiv(OC, meta['BLOCK_N']))

        conv_fused_kernel[grid](
            x, w, b_conv, b_extra, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            self.constant_value, self.scaling_factor,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out