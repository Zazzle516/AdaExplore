import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def convt2d_nhwc_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    OHOW = OH * OW
    m_mask = offs_m < OHOW
    n_mask = offs_n < OC

    oh = offs_m // OW
    ow = offs_m % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        ih_valid = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW)
            valid = ih_valid & iw_valid

            ih_safe = tl.where(valid, ih, 0)
            iw_safe = tl.where(valid, iw, 0)
            x_row_base = pid_b * IH * IW * IC + ih_safe * IW * IC + iw_safe * IC

            w_kh_kw_base = kh * KW * OC + kw * OC

            for k in range(0, IC, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                k_mask = offs_k < IC

                x_ptrs = x_ptr + x_row_base[:, None] + offs_k[None, :]
                x_full_mask = (valid & m_mask)[:, None] & k_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_full_mask, other=0.0)

                w_ptrs = w_ptr + offs_k[:, None] * (KH * KW * OC) + w_kh_kw_base + offs_n[None, :]
                w_full_mask = k_mask[:, None] & n_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_full_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += b[None, :]

    out_row_base = pid_b * OHOW * OC + offs_m * OC
    out_ptrs = out_ptr + out_row_base[:, None] + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def convt2d_nhwc_fused_kernel(
    x_ptr, w_ptr, b_ptr, bias2_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    scaling_factor,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # This kernel computes conv-transpose AND the softmax+bias+scale+sigmoid epilogue
    # in one shot. Assumes BLOCK_N >= OC (single OC tile), so softmax can be done in-register.
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    OHOW = OH * OW
    m_mask = offs_m < OHOW
    n_mask = offs_n < OC

    oh = offs_m // OW
    ow = offs_m % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        ih_valid = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW)
            valid = ih_valid & iw_valid

            ih_safe = tl.where(valid, ih, 0)
            iw_safe = tl.where(valid, iw, 0)
            x_row_base = pid_b * IH * IW * IC + ih_safe * IW * IC + iw_safe * IC

            w_kh_kw_base = kh * KW * OC + kw * OC

            for k in range(0, IC, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                k_mask = offs_k < IC

                x_ptrs = x_ptr + x_row_base[:, None] + offs_k[None, :]
                x_full_mask = (valid & m_mask)[:, None] & k_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_full_mask, other=0.0)

                w_ptrs = w_ptr + offs_k[:, None] * (KH * KW * OC) + w_kh_kw_base + offs_n[None, :]
                w_full_mask = k_mask[:, None] & n_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_full_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Add conv bias
    b_conv = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += b_conv[None, :]

    # Softmax across channels (axis=1 of acc, the BLOCK_N dim) per row
    neg_inf = float('-inf')
    acc_masked = tl.where(n_mask[None, :], acc, neg_inf)
    row_max = tl.max(acc_masked, axis=1)  # [BLOCK_M]
    e = tl.exp(acc - row_max[:, None])
    e = tl.where(n_mask[None, :], e, 0.0)
    row_sum = tl.sum(e, axis=1)  # [BLOCK_M]
    sm = e / row_sum[:, None]

    # bias2 + scale + sigmoid
    bias2 = tl.load(bias2_ptr + offs_n, mask=n_mask, other=0.0)
    y = (sm + bias2[None, :]) * scaling_factor
    out = 1.0 / (1.0 + tl.exp(-y))

    out_row_base = pid_b * OHOW * OC + offs_m * OC
    out_ptrs = out_ptr + out_row_base[:, None] + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        stride = self.stride
        pad = self.padding
        opad = self.output_padding

        OH = (IH - 1) * stride - 2 * pad + KH + opad
        OW = (IW - 1) * stride - 2 * pad + KW + opad

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w = self.conv_transpose.weight  # [IC, OC, KH, KW]
        w_reshaped = w.permute(0, 2, 3, 1).contiguous()  # [IC, KH, KW, OC]

        conv_bias = self.conv_transpose.bias
        if conv_bias is None:
            conv_bias = torch.zeros(OC, device=x.device, dtype=torch.float32)
        conv_bias = conv_bias.contiguous()

        bias2_flat = self.bias.view(-1).contiguous()

        out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=torch.float32)

        # BLOCK_N must cover OC so softmax is in-register
        BLOCK_N = triton.next_power_of_2(OC)
        BLOCK_M = 64
        BLOCK_K = 32

        OHOW = OH * OW
        grid = (triton.cdiv(OHOW, BLOCK_M), N)

        convt2d_nhwc_fused_kernel[grid](
            x_nhwc, w_reshaped, conv_bias, bias2_flat, out_nhwc,
            N, IC, IH, IW,
            OC, OH, OW,
            float(self.scaling_factor),
            KH, KW,
            stride, pad,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out