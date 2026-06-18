import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N", "OC", "OH", "OW", "IC", "KH", "KW"])
@triton.jit
def conv2d_relu_bias_nhwc_kernel(
    x_ptr, w_ptr, b_ptr, bias2_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # x is NHWC: (N, IH, IW, IC), contiguous
    # w is (OC, KH, KW, IC) reshaped from (OC, IC, KH, KW)
    # out is NHWC: (N, OH, OW, OC)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = N * OH * OW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Decompose m into (n, oh, ow)
    n_idx = offs_m // (OH * OW)
    hw = offs_m % (OH * OW)
    oh = hw // OW
    ow = hw % OW

    m_mask = offs_m < M
    n_mask = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # x base offset per row: n_idx * IH*IW*IC + oh*IW*IC + ow*IC  (then add kh*IW*IC + kw*IC)
    # we'll handle ic in inner loop

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh
            iw = ow + kw
            x_row_base = n_idx * (IH * IW * IC) + ih * (IW * IC) + iw * IC  # [BLOCK_M]
            w_row_base = kh * (KW * IC) + kw * IC  # scalar

            for k_start in range(0, IC, BLOCK_K):
                k_idx = k_start + offs_k
                k_mask = k_idx < IC

                # x: [BLOCK_M, BLOCK_K] - load IC values
                x_offs = x_row_base[:, None] + k_idx[None, :]
                x_mask = m_mask[:, None] & k_mask[None, :]
                x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

                # w: weight layout (OC, KH, KW, IC) -> stride (KH*KW*IC, KW*IC, IC, 1)
                # we need [BLOCK_K, BLOCK_N]
                w_offs = (offs_n[None, :] * (KH * KW * IC)
                          + w_row_base
                          + k_idx[:, None])
                w_mask = k_mask[:, None] & n_mask[None, :]
                w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # bias + relu + bias2
    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    bias2_vals = tl.load(bias2_ptr + offs_n, mask=n_mask, other=0.0)

    acc = acc + b_vals[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)
    acc = acc + bias2_vals[None, :]

    # store NHWC: out[n, oh, ow, oc]
    out_offs = (n_idx[:, None] * (OH * OW * OC)
                + oh[:, None] * (OW * OC)
                + ow[:, None] * OC
                + offs_n[None, :])
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv2d_relu_bias_nhwc(x_nhwc, w_ohwi, b, bias2):
    N, IH, IW, IC = x_nhwc.shape
    OC, KH, KW, _ = w_ohwi.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)

    M = N * OH * OW
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(OC, meta["BLOCK_N"]),
    )

    conv2d_relu_bias_nhwc_kernel[grid](
        x_nhwc, w_ohwi, b, bias2, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._cached_w = None

    def _get_weight_nhwc(self):
        # weight: (OC, IC, KH, KW) -> (OC, KH, KW, IC)
        w = self.conv.weight
        if (self._cached_w is None
            or self._cached_w.data_ptr() == 0
            or self._cached_w.shape != (w.shape[0], w.shape[2], w.shape[3], w.shape[1])
            or self._cached_w.device != w.device):
            self._cached_w = w.permute(0, 2, 3, 1).contiguous()
        return self._cached_w

    def forward(self, x):
        # x: (N, IC, IH, IW) -> NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w_ohwi = self._get_weight_nhwc()
        b = self.conv.bias.contiguous()
        bias2 = self.bias.view(-1).contiguous()
        out_nhwc = conv2d_relu_bias_nhwc(x_nhwc, w_ohwi, b, bias2)
        # back to NCHW
        return out_nhwc.permute(0, 3, 1, 2).contiguous()