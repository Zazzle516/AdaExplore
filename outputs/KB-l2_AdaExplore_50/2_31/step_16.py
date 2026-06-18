import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _conv_configs():
    configs = []
    for bm, bn, bk, nw, ns in [
        (64, 64, 32, 4, 2),
        (64, 128, 16, 4, 2),
        (128, 64, 16, 4, 2),
        (64, 64, 16, 4, 2),
        (32, 64, 16, 4, 2),
        (64, 32, 16, 4, 2),
        (128, 64, 32, 8, 2),
        (64, 128, 32, 8, 2),
    ]:
        configs.append(triton.Config(
            {'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': bk},
            num_warps=nw, num_stages=ns
        ))
    return configs


@triton.autotune(configs=_conv_configs(), key=['N', 'OC', 'IC', 'OH', 'OW', 'KH', 'KW'])
@triton.jit
def _conv2d_fused_kernel(
    x_ptr, w_ptr, cb_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    constant_value, scaling_factor,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # over N * (OH*OW) tiles
    pid_n = tl.program_id(1)  # over OC tiles

    OHW = OH * OW
    M_total = N * OHW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N] - OC indices

    mask_m = offs_m < M_total
    mask_n = offs_n < OC

    n_idx = offs_m // OHW
    hw_idx = offs_m % OHW
    oh_idx = hw_idx // OW
    ow_idx = hw_idx % OW

    K_total = IC * KH * KW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    for k_start in range(0, K_total, BLOCK_K):
        k = k_start + offs_k  # [BLOCK_K]
        mask_k = k < K_total

        ic = k // (KH * KW)
        khw = k % (KH * KW)
        kh = khw // KW
        kw = khw % KW

        # Load x: [BLOCK_M, BLOCK_K]
        ih = oh_idx[:, None] + kh[None, :]
        iw = ow_idx[:, None] + kw[None, :]
        x_off = (n_idx[:, None] * stride_xn +
                 ic[None, :] * stride_xc +
                 ih * stride_xh +
                 iw * stride_xw)
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        # Load w: [BLOCK_K, BLOCK_N]
        w_off = (offs_n[None, :] * stride_wo +
                 ic[:, None] * stride_wi +
                 kh[:, None] * stride_wkh +
                 kw[:, None] * stride_wkw)
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # Add conv bias
    cb = tl.load(cb_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + cb[None, :]

    # Min with constant
    acc = tl.minimum(acc, constant_value)

    # Add per-channel bias (shape: OC)
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # Scale
    acc = acc * scaling_factor

    # Store
    out_off = (n_idx[:, None] * stride_on +
               offs_n[None, :] * stride_oc +
               oh_idx[:, None] * stride_oh +
               ow_idx[:, None] * stride_ow)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def fused_conv2d(x, weight, conv_bias, bias, constant_value, scaling_factor):
    x = x.contiguous()
    weight = weight.contiguous()
    N, IC, IH, IW = x.shape
    OC, ICw, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    bias_flat = bias.contiguous().view(-1)
    cb_flat = conv_bias.contiguous().view(-1)

    grid = lambda meta: (
        triton.cdiv(N * OH * OW, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    _conv2d_fused_kernel[grid](
        x, weight, cb_flat, bias_flat, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        float(constant_value), float(scaling_factor),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
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

    def forward(self, x):
        return fused_conv2d(
            x, self.conv.weight, self.conv.bias,
            self.bias, self.constant_value, self.scaling_factor
        )