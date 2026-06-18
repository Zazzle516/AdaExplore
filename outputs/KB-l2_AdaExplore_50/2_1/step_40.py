import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N", "OC", "OH", "OW", "IC", "KH", "KW"])
@triton.jit
def conv2d_relu_bias_kernel_nhwc(
    x_ptr, w_ptr, b_ptr, bias2_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_wkhkwi, stride_wo,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,  # output spatial tile (OH*OW)
    BLOCK_N: tl.constexpr,  # OC tile
    BLOCK_K: tl.constexpr,  # IC reduction tile
):
    pid_n = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # output spatial tile id
    pid_oc = tl.program_id(2)  # OC tile id

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    oh = offs_m // OW
    ow = offs_m % OW

    m_mask = offs_m < (OH * OW)
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_base = x_ptr + pid_n * stride_xn
    # spatial offset for each output position (input top-left of receptive field)
    spatial_off = oh * stride_xh + ow * stride_xw  # [BLOCK_M]

    # Flatten K = KH*KW*IC into one dimension. Inside each K-tile (BLOCK_K),
    # all K elements share the same kh,kw if BLOCK_K <= IC and IC % BLOCK_K == 0,
    # which holds here (IC=64, BLOCK_K in {16,32}). We rely on that to
    # compute kh, kw once per K-tile.
    K_TOTAL = KH * KW * IC

    for k_start in range(0, K_TOTAL, BLOCK_K):
        # k_start is the flattened K offset; decompose
        khkw = k_start // IC
        kh = khkw // KW
        kw = khkw % KW
        ic_start = k_start - khkw * IC
        ic = ic_start + offs_k  # [BLOCK_K]
        k_mask = ic < IC

        kh_kw_off = kh * stride_xh + kw * stride_xw

        # x: [BLOCK_M, BLOCK_K]
        x_offs = (spatial_off[:, None]
                  + kh_kw_off
                  + ic[None, :] * stride_xc)
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_base + x_offs, mask=x_mask, other=0.0)

        # w: [BLOCK_K, BLOCK_N], weights laid out as (KH*KW*IC, OC)
        # contiguous along OC, K stride = stride_wkhkwi (= OC)
        k_flat = k_start + offs_k
        k_flat_mask = k_flat < K_TOTAL
        w_offs = (k_flat[:, None] * stride_wkhkwi
                  + offs_n[None, :] * stride_wo)
        w_mask = k_flat_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    bias2_vals = tl.load(bias2_ptr + offs_n, mask=n_mask, other=0.0)

    acc = acc + b_vals[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)
    acc = acc + bias2_vals[None, :]

    # store NCHW output: out[N, OC, OH, OW]
    out_offs = (pid_n * stride_on
                + offs_n[None, :] * stride_oc
                + oh[:, None] * stride_oh
                + ow[:, None] * stride_ow)
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv2d_relu_bias(x_nhwc, w_kkic_oc, b, bias2, N, IC, IH, IW, OC, KH, KW):
    OH = IH - KH + 1
    OW = IW - KW + 1

    # Output directly in NCHW layout
    out_nchw = torch.empty((N, OC, OH, OW), device=x_nhwc.device, dtype=x_nhwc.dtype)

    grid = lambda meta: (
        N,
        triton.cdiv(OH * OW, meta["BLOCK_M"]),
        triton.cdiv(OC, meta["BLOCK_N"]),
    )

    conv2d_relu_bias_kernel_nhwc[grid](
        x_nhwc, w_kkic_oc, b, bias2, out_nchw,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        w_kkic_oc.stride(0), w_kkic_oc.stride(1),
        out_nchw.stride(0), out_nchw.stride(1), out_nchw.stride(2), out_nchw.stride(3),
    )
    return out_nchw


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # x: (N, IC, IH, IW) -> NHWC
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        # weight: (OC, IC, KH, KW) -> (KH, KW, IC, OC) flattened to (KH*KW*IC, OC)
        w_kkic_oc = self.conv.weight.permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC)
        b = self.conv.bias.contiguous()
        bias2 = self.bias.view(-1).contiguous()

        out = conv2d_relu_bias(x_nhwc, w_kkic_oc, b, bias2, N, IC, IH, IW, OC, KH, KW)
        return out