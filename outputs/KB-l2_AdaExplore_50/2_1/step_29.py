import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N", "OC", "OH", "OW", "IC", "KH", "KW"])
@triton.jit
def conv2d_relu_bias_nhwc_kernel(
    x_ptr, w_ptr, b_ptr, bias2_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_wkh, stride_wkw, stride_wi, stride_wo,
    stride_on, stride_oh, stride_ow, stride_oc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # = IC, must be power-of-2 multiple
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_oc = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    oh = offs_m // OW
    ow = offs_m % OW

    m_mask = offs_m < (OH * OW)
    n_mask = offs_n < OC
    k_mask = offs_k < IC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_base = x_ptr + pid_n * stride_xn
    # spatial offset in NHWC layout (h*stride_xh + w*stride_xw)
    spatial_off = oh * stride_xh + ow * stride_xw  # [BLOCK_M]

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # x: [BLOCK_M, BLOCK_K=IC]
            x_offs = (spatial_off[:, None]
                      + kh * stride_xh
                      + kw * stride_xw
                      + offs_k[None, :] * stride_xc)
            x_mask = m_mask[:, None] & k_mask[None, :]
            x_vals = tl.load(x_base + x_offs, mask=x_mask, other=0.0)

            # w: [BLOCK_K=IC, BLOCK_N=OC tile]
            w_offs = (kh * stride_wkh
                      + kw * stride_wkw
                      + offs_k[:, None] * stride_wi
                      + offs_n[None, :] * stride_wo)
            w_mask = k_mask[:, None] & n_mask[None, :]
            w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

            acc += tl.dot(x_vals, w_vals)

    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    bias2_vals = tl.load(bias2_ptr + offs_n, mask=n_mask, other=0.0)

    acc = acc + b_vals[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)
    acc = acc + bias2_vals[None, :]

    out_offs = (pid_n * stride_on
                + oh[:, None] * stride_oh
                + ow[:, None] * stride_ow
                + offs_n[None, :] * stride_oc)
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def conv2d_relu_bias_nhwc(x_nhwc, w_kh_kw_ic_oc, b, bias2, OC, IC, KH, KW, OH, OW, N):
    out = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)
    BLOCK_K = _next_pow2(IC)

    grid = lambda meta: (
        N,
        triton.cdiv(OH * OW, meta["BLOCK_M"]),
        triton.cdiv(OC, meta["BLOCK_N"]),
    )

    conv2d_relu_bias_nhwc_kernel[grid](
        x_nhwc, w_kh_kw_ic_oc, b, bias2, out,
        N, IC, x_nhwc.shape[1], x_nhwc.shape[2],
        OC, OH, OW,
        KH, KW,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        w_kh_kw_ic_oc.stride(0), w_kh_kw_ic_oc.stride(1), w_kh_kw_ic_oc.stride(2), w_kh_kw_ic_oc.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_K=BLOCK_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._cached_w_nhwc = None
        self._cached_w_id = None

    def _get_weight_nhwc(self):
        w = self.conv.weight
        wid = (w.data_ptr(), w.shape, w._version)
        if self._cached_w_id != wid:
            # weight: [OC, IC, KH, KW] -> [KH, KW, IC, OC]
            w_perm = w.detach().permute(2, 3, 1, 0).contiguous()
            self._cached_w_nhwc = w_perm
            self._cached_w_id = wid
        return self._cached_w_nhwc

    def forward(self, x):
        N, IC, IH, IW = x.shape
        OC = self.conv.out_channels
        KH, KW = self.conv.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w_nhwc = self._get_weight_nhwc()
        b = self.conv.bias.contiguous()
        bias2 = self.bias.view(-1).contiguous()

        out_nhwc = conv2d_relu_bias_nhwc(
            x_nhwc, w_nhwc, b, bias2, OC, IC, KH, KW, OH, OW, N
        )
        # Convert back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out