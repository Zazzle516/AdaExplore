import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_fused_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    OUT_HW,
    constant_value, scaling_factor,
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_wo, stride_wh, stride_ww, stride_wi,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # x is NHWC, w is OC,KH,KW,IC (both channels-last layout for IC contiguous)
    pid_m = tl.program_id(0)  # OC tile
    pid_n = tl.program_id(1)  # spatial tile
    pid_b = tl.program_id(2)  # batch

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial (oh*OW + ow)
    offs_k = tl.arange(0, BLOCK_K)  # IC

    mask_m = offs_m < OC
    mask_n = offs_n < OUT_HW

    oh = offs_n // OW
    ow = offs_n % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kh, kw, then ic-block
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # [BLOCK_N]
            iw = ow + kw  # [BLOCK_N]
            # x base for this (kh,kw): pointer to x[pid_b, ih, iw, 0]
            x_base = pid_b * stride_xn + ih * stride_xh + iw * stride_xw  # [BLOCK_N]
            # w base for this (kh,kw): pointer to w[offs_m, kh, kw, 0]
            w_base = offs_m * stride_wo + kh * stride_wh + kw * stride_ww  # [BLOCK_M]

            for ic_start in range(0, IC, BLOCK_K):
                ic_idx = ic_start + offs_k  # [BLOCK_K]
                ic_mask = ic_idx < IC

                # Load weight: [BLOCK_M, BLOCK_K], contiguous along IC
                w_offset = w_base[:, None] + ic_idx[None, :] * stride_wi
                w_mask = mask_m[:, None] & ic_mask[None, :]
                w_vals = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

                # Load input: [BLOCK_K, BLOCK_N], contiguous along IC
                x_offset = x_base[None, :] + ic_idx[:, None] * stride_xc
                x_mask = (ic_mask[:, None]
                          & mask_n[None, :])
                x_vals = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

                acc += tl.dot(w_vals, x_vals, allow_tf32=True, out_dtype=tl.float32)

    # add conv bias (per-OC)
    b_vals = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + b_vals[:, None]

    # min with constant
    acc = tl.minimum(acc, constant_value)

    # add extra bias (per-OC)
    bias_vals = tl.load(bias_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + bias_vals[:, None]

    # scale
    acc = acc * scaling_factor

    # store: out is NCHW
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
        # Convert to NHWC for coalesced IC loads
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [N, H, W, IC]
        # Weight to OC, KH, KW, IC
        w_ohwi = self.conv.weight.permute(0, 2, 3, 1).contiguous()  # [OC, KH, KW, IC]
        b = self.conv.bias.contiguous()
        bias = self.bias.contiguous().view(-1)

        N, H, W, IC = x_nhwc.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        OUT_HW = OH * OW

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda META: (
            triton.cdiv(OC, META['BLOCK_M']),
            triton.cdiv(OUT_HW, META['BLOCK_N']),
            N,
        )

        conv2d_fused_kernel[grid](
            x_nhwc, w_ohwi, b, bias, out,
            N, IC, H, W,
            OC, OH, OW,
            KH, KW,
            OUT_HW,
            float(self.constant_value), float(self.scaling_factor),
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            w_ohwi.stride(0), w_ohwi.stride(1), w_ohwi.stride(2), w_ohwi.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        )
        return out