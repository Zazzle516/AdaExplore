import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["N", "OC", "OH", "OW", "IC", "KH", "KW"])
@triton.jit
def conv2d_relu_bias_kernel(
    x_ptr, w_ptr, b_ptr, bias2_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_oc = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    oh = offs_m // OW
    ow = offs_m % OW

    K = IC * KH * KW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_base = x_ptr + pid_n * stride_xn
    spatial_off = oh * stride_xh + ow * stride_xw
    m_mask = offs_m < (OH * OW)
    n_mask = offs_n < OC

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        k_mask = k_idx < K

        ic = k_idx // (KH * KW)
        rem = k_idx % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        kh_kw_off = kh * stride_xh + kw * stride_xw
        x_offs = (spatial_off[:, None]
                  + ic[None, :] * stride_xc
                  + kh_kw_off[None, :])

        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_base + x_offs, mask=x_mask, other=0.0)

        w_offs = (offs_n[None, :] * stride_wo
                  + ic[:, None] * stride_wi
                  + kh[:, None] * stride_wkh
                  + kw[:, None] * stride_wkw)
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    bias2_vals = tl.load(bias2_ptr + offs_n, mask=n_mask, other=0.0)

    acc = acc + b_vals[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)
    acc = acc + bias2_vals[None, :]

    out_offs = (pid_n * stride_on
                + offs_n[None, :] * stride_oc
                + oh[:, None] * stride_oh
                + ow[:, None] * stride_ow)
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def conv2d_relu_bias(x, w, b, bias2):
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    bias2 = bias2.contiguous()

    N, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        N,
        triton.cdiv(OH * OW, meta["BLOCK_M"]),
        triton.cdiv(OC, meta["BLOCK_N"]),
    )

    conv2d_relu_bias_kernel[grid](
        x, w, b, bias2, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight
        b = self.conv.bias
        bias2 = self.bias.view(-1).contiguous()
        return conv2d_relu_bias(x, w, b, bias2)