import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Conv2d implemented as implicit-im2col GEMM in NHWC layout, with fused epilogue:
#   y = (min(conv(x) + conv_bias, c) + extra_bias) * s
#
# Layout:
#   x: (N, H, W, IC)      contiguous NHWC
#   w: (KH*KW*IC, OC)     pre-permuted weights for clean K-loop
#   y: (N, OH, OW, OC)    contiguous NHWC
#
# GEMM dims:
#   M = N * OH * OW  (rows)
#   N_dim = OC       (cols)
#   K = KH * KW * IC


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 16}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K", "OH", "OW", "IC", "KH", "KW"])
@triton.jit
def conv2d_implicit_gemm_kernel(
    x_ptr, w_ptr, conv_bias_ptr, extra_bias_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    KH, KW,
    M, N_dim, K,
    constant_value, scaling_factor,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row in M = N*OH*OW
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # col in OC
    offs_k = tl.arange(0, BLOCK_K)

    # Decompose row index into (n, oh, ow)
    OHW = OH * OW
    n_idx = offs_m // OHW
    rem = offs_m % OHW
    oh_idx = rem // OW
    ow_idx = rem % OW

    m_mask = offs_m < M
    n_mask = offs_n < N_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = KH*KW*IC. Iterate over K in BLOCK_K chunks.
    KIC = KW * IC
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K

        # Decompose k -> (kh, kw, ic)
        kh = k_idx // KIC
        krem = k_idx % KIC
        kw = krem // IC
        ic = krem % IC

        # input h, w positions
        ih = oh_idx[:, None] + kh[None, :]  # [BLOCK_M, BLOCK_K]
        iw = ow_idx[:, None] + kw[None, :]

        # x_ptr offsets: NHWC
        x_offs = (
            n_idx[:, None] * stride_xn
            + ih * stride_xh
            + iw * stride_xw
            + ic[None, :] * stride_xc
        )
        x_load_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptr + x_offs, mask=x_load_mask, other=0.0)

        # weight offsets: w is (K, OC), row-major contiguous
        w_offs = k_idx[:, None] * N_dim + offs_n[None, :]
        w_load_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_ptr + w_offs, mask=w_load_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # Epilogue: + conv_bias[oc], min(., constant), + extra_bias[oc], * scaling
    cb = tl.load(conv_bias_ptr + offs_n, mask=n_mask, other=0.0)
    eb = tl.load(extra_bias_ptr + offs_n, mask=n_mask, other=0.0)

    acc = acc + cb[None, :]
    acc = tl.minimum(acc, constant_value)
    acc = acc + eb[None, :]
    acc = acc * scaling_factor

    # Store to NHWC output: y[n, oh, ow, oc]
    y_offs = (
        n_idx[:, None] * stride_yn
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
        + offs_n[None, :] * stride_yc
    )
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + y_offs, acc, mask=store_mask)


def fused_conv2d(x_nchw, weight, conv_bias, extra_bias, constant_value, scaling_factor):
    """
    x_nchw: (N, IC, H, W)
    weight: (OC, IC, KH, KW)
    conv_bias: (OC,)
    extra_bias: (OC,) flattened
    Returns: (N, OC, OH, OW) NCHW directly
    """
    N, IC, H, W = x_nchw.shape
    OC, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    x_nchw = x_nchw.contiguous()

    # Permute weight to (KH, KW, IC, OC) -> flatten to (KH*KW*IC, OC)
    w_perm = weight.permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC)

    out_nchw = torch.empty((N, OC, OH, OW), device=x_nchw.device, dtype=x_nchw.dtype)

    M = N * OH * OW
    N_dim = OC
    K = KH * KW * IC

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N_dim, meta["BLOCK_N"]),
    )

    conv2d_implicit_gemm_kernel[grid](
        x_nchw, w_perm, conv_bias, extra_bias, out_nchw,
        N, IC, H, W,
        OC, OH, OW,
        KH, KW,
        M, N_dim, K,
        float(constant_value), float(scaling_factor),
        x_nchw.stride(0), x_nchw.stride(1), x_nchw.stride(2), x_nchw.stride(3),
        out_nchw.stride(0), out_nchw.stride(1), out_nchw.stride(2), out_nchw.stride(3),
    )

    return out_nchw


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
        x = x.cuda() if not x.is_cuda else x
        weight = self.conv.weight
        conv_bias = self.conv.bias
        if conv_bias is None:
            conv_bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        extra_bias = self.bias.view(-1).contiguous()
        out = fused_conv2d(
            x, weight, conv_bias.contiguous(), extra_bias,
            self.constant_value, self.scaling_factor,
        )
        return out