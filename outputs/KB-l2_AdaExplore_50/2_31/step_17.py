import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Conv2d with implicit im2col GEMM in NHWC layout, fused epilogue:
# out = (min(conv(x) + conv_bias, const) + extra_bias) * scale
# Layout:
#   x_nhwc: (N, H, W, IC)
#   weight_packed: (KH*KW*IC, OC)  (row-major)
#   out_nhwc: (N, OH, OW, OC)

def _conv_configs():
    return [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ]


@triton.autotune(configs=_conv_configs(), key=['N', 'IC', 'OC', 'H', 'W', 'KH', 'KW'])
@triton.jit
def conv2d_nhwc_fused_kernel(
    x_ptr, w_ptr, cb_ptr, eb_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    constant_value, scaling_factor,
    # strides for x (NHWC contiguous): N*H*W*IC, H*W*IC, W*IC, IC, 1 not all needed
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_on, stride_oh, stride_ow, stride_oc,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile (n*OH*OW)
    BLOCK_K: tl.constexpr,  # reduction tile
):
    pid_m = tl.program_id(0)  # OC dim
    pid_n = tl.program_id(1)  # spatial (N*OH*OW) dim

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC offsets
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial offsets

    # Decode spatial offsets into (n, oh, ow)
    OHW = OH * OW
    n_idx = offs_n // OHW
    rem = offs_n % OHW
    oh_idx = rem // OW
    ow_idx = rem % OW

    mask_m = offs_m < OC
    mask_n = offs_n < (N * OHW)

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    K_total = KH * KW * IC

    # Loop over K (the unfolded patch dimension)
    for k_start in range(0, K_total, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offs < K_total

        # decode k -> kh, kw, ic
        kh = k_offs // (KW * IC)
        rem_k = k_offs % (KW * IC)
        kw = rem_k // IC
        ic = rem_k % IC

        # input spatial indices
        ih = oh_idx[:, None] + kh[None, :]  # (BLOCK_N, BLOCK_K)
        iw = ow_idx[:, None] + kw[None, :]

        # input pointer
        x_off = (n_idx[:, None] * stride_xn +
                 ih * stride_xh +
                 iw * stride_xw +
                 ic[None, :] * stride_xc)

        x_mask = mask_n[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # (BLOCK_N, BLOCK_K)

        # weight: (K_total, OC), row-major. w[k, m]
        w_off = k_offs[:, None] * OC + offs_m[None, :]
        w_mask = mask_k[:, None] & mask_m[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # (BLOCK_K, BLOCK_M)

        acc += tl.dot(x_tile, w_tile, allow_tf32=False)

    # epilogue: add conv bias, min, add extra bias, scale
    cb = tl.load(cb_ptr + offs_m, mask=mask_m, other=0.0)  # (BLOCK_M,)
    eb = tl.load(eb_ptr + offs_m, mask=mask_m, other=0.0)

    acc = acc + cb[None, :]
    acc = tl.minimum(acc, constant_value)
    acc = acc + eb[None, :]
    acc = acc * scaling_factor

    # store
    out_off = (n_idx[:, None] * stride_on +
               oh_idx[:, None] * stride_oh +
               ow_idx[:, None] * stride_ow +
               offs_m[None, :] * stride_oc)
    out_mask = mask_n[:, None] & mask_m[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv2d_fused(x, weight, conv_bias, extra_bias, constant_value, scaling_factor):
    """
    x: (N, IC, H, W) contiguous fp32
    weight: (OC, IC, KH, KW)
    conv_bias: (OC,)
    extra_bias: (OC, 1, 1) or shape broadcastable to (OC,)
    Returns (N, OC, OH, OW) in NCHW layout (contiguous).
    """
    N, IC, H, W = x.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    # NHWC input
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, IC)

    # weight pack: (OC, IC, KH, KW) -> (KH, KW, IC, OC) -> (KH*KW*IC, OC)
    w_packed = weight.permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC)

    out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

    cb = conv_bias.contiguous().view(-1)
    eb = extra_bias.contiguous().view(-1)

    # strides
    stride_xn = H * W * IC
    stride_xh = W * IC
    stride_xw = IC
    stride_xc = 1
    stride_on = OH * OW * OC
    stride_oh = OW * OC
    stride_ow = OC
    stride_oc = 1

    grid = lambda META: (
        triton.cdiv(OC, META['BLOCK_M']),
        triton.cdiv(N * OH * OW, META['BLOCK_N']),
    )

    conv2d_nhwc_fused_kernel[grid](
        x_nhwc, w_packed, cb, eb, out_nhwc,
        N, IC, H, W,
        OC, OH, OW,
        KH, KW,
        float(constant_value), float(scaling_factor),
        stride_xn, stride_xh, stride_xw, stride_xc,
        stride_on, stride_oh, stride_ow, stride_oc,
    )

    # convert back to NCHW
    out = out_nhwc.permute(0, 3, 1, 2).contiguous()
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = x.contiguous()
        return conv2d_fused(
            x,
            self.conv.weight,
            self.conv.bias,
            self.bias,
            self.constant_value,
            self.scaling_factor,
        )