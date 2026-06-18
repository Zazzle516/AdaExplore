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
    pid_b = tl.program_id(2)  # batch index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial idx within batch (oh*OW+ow)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC

    OHOW = OH * OW
    m_mask = offs_m < OHOW
    n_mask = offs_n < OC

    oh = offs_m // OW
    ow = offs_m % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For each kernel position (kh, kw), determine if there's a valid input pixel
    # ih*STRIDE - PAD + kh = oh  =>  ih = (oh + PAD - kh) / STRIDE
    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        ih_valid = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW)
            valid = ih_valid & iw_valid  # [BLOCK_M]

            # Pointers to input: x is [N, IH, IW, IC] (NHWC)
            ih_safe = tl.where(valid, ih, 0)
            iw_safe = tl.where(valid, iw, 0)
            x_row_base = pid_b * IH * IW * IC + ih_safe * IW * IC + iw_safe * IC  # [BLOCK_M]

            # Weight is [IC, OC, KH, KW] in original layout. We reshape to [IC, KH, KW, OC]
            # so that for fixed (kh, kw), w[:, kh, kw, :] is contiguous in OC for each IC.
            # w_ptr offset for (ic, kh, kw, oc) = ic * (KH*KW*OC) + kh * (KW*OC) + kw * OC + oc
            w_kh_kw_base = kh * KW * OC + kw * OC

            # GEMM loop over IC
            for k in range(0, IC, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                k_mask = offs_k < IC

                # Load x: [BLOCK_M, BLOCK_K]
                x_ptrs = x_ptr + x_row_base[:, None] + offs_k[None, :]
                x_full_mask = (valid & m_mask)[:, None] & k_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_full_mask, other=0.0)

                # Load w: [BLOCK_K, BLOCK_N]
                # w[ic, kh, kw, oc] = w_ptr + ic*(KH*KW*OC) + kh*KW*OC + kw*OC + oc
                w_ptrs = w_ptr + offs_k[:, None] * (KH * KW * OC) + w_kh_kw_base + offs_n[None, :]
                w_full_mask = k_mask[:, None] & n_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_full_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Add bias
    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += b[None, :]

    # Output: NHWC layout [N, OH, OW, OC]
    out_row_base = pid_b * OHOW * OC + offs_m * OC  # [BLOCK_M]
    out_ptrs = out_ptr + out_row_base[:, None] + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def fused_softmax_bias_scale_sigmoid_nhwc_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, HW,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    # x is NHWC: [N, H, W, C]. One program per (n, hw) — channels are contiguous.
    pid = tl.program_id(0)
    n = pid // HW
    hw = pid % HW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = n * HW * C + hw * C
    x_ptrs = x_ptr + base + offs_c

    x = tl.load(x_ptrs, mask=mask_c, other=-float('inf'))
    x_f = x.to(tl.float32)

    max_val = tl.max(x_f, axis=0)
    e = tl.exp(x_f - max_val)
    e = tl.where(mask_c, e, 0.0)
    sum_e = tl.sum(e, axis=0)
    sm = e / sum_e

    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    y = (sm + b) * scaling_factor
    out = 1.0 / (1.0 + tl.exp(-y))

    out_ptrs = out_ptr + base + offs_c
    tl.store(out_ptrs, out, mask=mask_c)


def custom_convt2d(x_nhwc, w_reshaped, bias, N, IC, IH, IW, OC, OH, OW, KH, KW, stride, pad):
    out = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32

    OHOW = OH * OW
    grid = (triton.cdiv(OHOW, BLOCK_M), triton.cdiv(OC, BLOCK_N), N)

    convt2d_nhwc_kernel[grid](
        x_nhwc, w_reshaped, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, pad,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return out


def fused_epilogue_nhwc(x_nhwc, bias_flat, scaling_factor):
    N, H, W, C = x_nhwc.shape
    HW = H * W
    out = torch.empty_like(x_nhwc)
    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * HW,)
    fused_softmax_bias_scale_sigmoid_nhwc_kernel[grid](
        x_nhwc, bias_flat, out,
        N, C, HW,
        float(scaling_factor),
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


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

        # Convert input to NHWC contiguous
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        # Weight: [IC, OC, KH, KW] -> [IC, KH, KW, OC]
        w = self.conv_transpose.weight  # [IC, OC, KH, KW]
        w_reshaped = w.permute(0, 2, 3, 1).contiguous()

        conv_bias = self.conv_transpose.bias
        if conv_bias is None:
            conv_bias = torch.zeros(OC, device=x.device, dtype=torch.float32)

        # Run custom convT — only supports cases where OH/OW match the formula AND output_padding is "absorbed".
        # Our index math: ih = (oh + pad - kh)/stride. With output_padding=1 stride=2 pad=1 KH=4,
        # output extends to OH = (IH-1)*2 - 2 + 4 + 1 = 2*IH+1. For ow=OW-1 (last col), iw_num = OW-1+1-kw = 2*IW-kw.
        # iw = (2*IW - kw)/2 — for kw=0: iw=IW (out of range, valid=False); kw=2: iw=IW-1 OK. So output_padding handled
        # naturally by the bounds check on ih/iw.
        out_nhwc = custom_convt2d(
            x_nhwc, w_reshaped, conv_bias.contiguous(),
            N, IC, IH, IW, OC, OH, OW, KH, KW, stride, pad,
        )

        # Fused epilogue: softmax over channels (which are contiguous in NHWC), add bias, scale, sigmoid
        bias_flat = self.bias.view(-1).contiguous()
        out_nhwc = fused_epilogue_nhwc(out_nhwc, bias_flat, self.scaling_factor)

        # Convert back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out