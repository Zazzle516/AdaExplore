import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _conv_configs():
    return [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ]


@triton.autotune(configs=_conv_configs(), key=['OC', 'OHW', 'IC_KK'])
@triton.jit
def _conv_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, extra_bias_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    OHW, IC_KK,
    constant_value, scaling_factor,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch index
    pid_oc = tl.program_id(1)  # OC tile
    pid_ohw = tl.program_id(2)  # OHW tile

    offs_oc = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_ohw = pid_ohw * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    oh = offs_ohw // OW
    ow = offs_ohw % OW

    mask_oc = offs_oc < OC
    mask_ohw = offs_ohw < OHW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over IC*KH*KW
    offs_k = tl.arange(0, BLOCK_K)
    for k_start in range(0, IC_KK, BLOCK_K):
        k = k_start + offs_k  # [BK]
        mask_k = k < IC_KK

        ic = k // (KH * KW)
        rem = k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # weight: [BM, BK] = w[oc, ic, kh, kw]
        w_offs = (offs_oc[:, None] * stride_wo +
                  ic[None, :] * stride_wi +
                  kh[None, :] * stride_wkh +
                  kw[None, :] * stride_wkw)
        w_mask = mask_oc[:, None] & mask_k[None, :]
        w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        # input: [BK, BN] = x[n, ic, oh+kh, ow+kw]
        ih = oh[None, :] + kh[:, None]
        iw = ow[None, :] + kw[:, None]
        x_offs = (pid_n * stride_xn +
                  ic[:, None] * stride_xc +
                  ih * stride_xh +
                  iw * stride_xw)
        x_mask = mask_k[:, None] & mask_ohw[None, :]
        x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

        acc += tl.dot(w_vals, x_vals)

    # add conv bias
    cb = tl.load(conv_bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + cb[:, None]

    # min
    acc = tl.minimum(acc, constant_value)

    # add extra bias (shape OC,1,1)
    eb = tl.load(extra_bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + eb[:, None]

    # scale
    acc = acc * scaling_factor

    # store
    out_offs = (pid_n * stride_on +
                offs_oc[:, None] * stride_oc +
                oh[None, :] * stride_oh +
                ow[None, :] * stride_ow)
    out_mask = mask_oc[:, None] & mask_ohw[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def fused_conv(x, weight, conv_bias, extra_bias, constant_value, scaling_factor):
    x = x.contiguous()
    weight = weight.contiguous()
    N, IC, H, W = x.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1
    OHW = OH * OW
    IC_KK = IC * KH * KW

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    extra_bias_flat = extra_bias.contiguous().view(-1)
    conv_bias_flat = conv_bias.contiguous().view(-1)

    grid = lambda meta: (
        N,
        triton.cdiv(OC, meta['BLOCK_M']),
        triton.cdiv(OHW, meta['BLOCK_N']),
    )

    _conv_fused_kernel[grid](
        x, weight, conv_bias_flat, extra_bias_flat, out,
        N, IC, H, W,
        OC, OH, OW,
        KH, KW,
        OHW, IC_KK,
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
        return fused_conv(
            x, self.conv.weight, self.conv.bias,
            self.bias, self.constant_value, self.scaling_factor,
        )