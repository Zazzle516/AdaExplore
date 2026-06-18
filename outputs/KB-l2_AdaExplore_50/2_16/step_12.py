import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_implicit_gemm_kernel(
    x_ptr,        # [N, IC, IH, IW]
    w_ptr,        # [OC, IC*KH*KW]  packed
    b_ptr,        # [OC]
    out_ptr,      # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    ADD_VAL: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    IC_CONST: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # spatial index into OH*OW
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # OC

    M = OH * OW
    m_mask = offs_m < M
    n_mask = offs_n < OC

    ow = offs_m % OW
    oh = offs_m // OW

    # Base input pointer for this batch
    x_base = pid_b * (IC * IH * IW)
    # Base output pointer for this batch
    out_base = pid_b * (OC * OH * OW)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_total = IC_CONST * KH * KW
    # iterate K in chunks of BLOCK_K
    offs_k_base = tl.arange(0, BLOCK_K)
    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + offs_k_base   # [BLOCK_K]
        k_mask = offs_k < K_total

        # decompose k -> (ic, kh, kw)
        kw_idx = offs_k % KW
        tmp = offs_k // KW
        kh_idx = tmp % KH
        ic_idx = tmp // KH

        # compute ih, iw for each (m, k)
        ih_num = oh[:, None] + PAD_H - kh_idx[None, :]   # [M, K]
        iw_num = ow[:, None] + PAD_W - kw_idx[None, :]
        ih = ih_num // STRIDE_H
        iw = iw_num // STRIDE_W
        ih_valid = ((ih_num % STRIDE_H) == 0) & (ih >= 0) & (ih < IH)
        iw_valid = ((iw_num % STRIDE_W) == 0) & (iw >= 0) & (iw < IW)
        spatial_valid = ih_valid & iw_valid & m_mask[:, None] & k_mask[None, :]

        x_off = x_base + ic_idx[None, :] * (IH * IW) + ih * IW + iw  # [M, K]
        a = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)    # [BLOCK_M, BLOCK_K]

        # Load B: weight [OC, K_total] -> [BLOCK_K, BLOCK_N]
        w_off = offs_k[:, None] * OC + offs_n[None, :]
        w_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(a, b)

    # Bias
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Mish: x * tanh(softplus(x))
    x = acc
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.where(x > 20.0, 0.0, x))))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = x * th
    y = y + ADD_VAL
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * SCALE

    out_off = out_base + offs_n[None, :] * (OH * OW) + offs_m[:, None]
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, y, mask=store_mask)


def conv_transpose_fused(x, weight_packed, bias, IC, OC, KH, KW,
                          stride, padding, output_padding, add_value, scale):
    x = x.contiguous()
    N, _, IH, IW = x.shape
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, OH, OW), dtype=x.dtype, device=x.device)

    M = OH * OW
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(OC, BLOCK_N), N)
    conv_transpose_implicit_gemm_kernel[grid](
        x, weight_packed, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, stride,
        padding, padding,
        float(add_value), float(scale),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        IC_CONST=IC,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-pack weight: original shape [IC, OC, KH, KW]
        # Target packed shape: [IC*KH*KW, OC] contiguous
        w = self.conv_transpose.weight.detach()  # [IC, OC, KH, KW]
        # permute to [IC, KH, KW, OC] -> reshape to [IC*KH*KW, OC]
        w_packed = w.permute(0, 2, 3, 1).contiguous().view(in_channels * kernel_size * kernel_size, out_channels)
        self.register_buffer('weight_packed', w_packed)
        self._weight_version = self.conv_transpose.weight._version

    def _maybe_repack(self):
        if self.training or self.conv_transpose.weight._version != self._weight_version:
            w = self.conv_transpose.weight.detach()
            w_packed = w.permute(0, 2, 3, 1).contiguous().view(
                self.in_channels * self.kernel_size * self.kernel_size, self.out_channels)
            self.weight_packed = w_packed
            self._weight_version = self.conv_transpose.weight._version

    def forward(self, x):
        self._maybe_repack()
        return conv_transpose_fused(
            x,
            self.weight_packed,
            self.conv_transpose.bias,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.add_value,
            self.scale,
        )