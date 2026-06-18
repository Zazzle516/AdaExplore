import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_mish_nhwc_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H_IN, W_IN,
    H_OUT, W_OUT,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    SUB: tl.constexpr,
    BLOCK_M: tl.constexpr,   # tile over output spatial (N*H_OUT*W_OUT)
    BLOCK_N: tl.constexpr,   # tile over C_OUT
    BLOCK_K: tl.constexpr,   # tile over K = C_IN
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    HW_OUT = H_OUT * W_OUT
    M = N * HW_OUT

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < C_OUT

    # decompose offs_m -> (n, oh, ow)
    n_idx = offs_m // HW_OUT
    rem = offs_m % HW_OUT
    oh = rem // W_OUT
    ow = rem % W_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < C_IN

    # x is NHWC: [N, H_IN, W_IN, C_IN], stride: (H_IN*W_IN*C_IN, W_IN*C_IN, C_IN, 1)
    # w is [C_OUT, KH, KW, C_IN] contiguous => for fixed (kh,kw,ic): w[oc, kh, kw, ic]
    # offset w: oc*KH*KW*C_IN + kh*KW*C_IN + kw*C_IN + ic
    x_stride_n = H_IN * W_IN * C_IN
    x_stride_h = W_IN * C_IN
    x_stride_w = C_IN

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh
            iw = ow + kw
            # base pointer per row m for this (kh,kw)
            x_base = n_idx * x_stride_n + ih * x_stride_h + iw * x_stride_w  # [BLOCK_M]
            # x tile: [BLOCK_M, BLOCK_K]
            x_offs = x_base[:, None] + offs_k[None, :]
            x_mask = mask_m[:, None] & mask_k[None, :]
            x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

            # w tile: [BLOCK_K, BLOCK_N]
            # w[oc, kh, kw, ic] - lay out as (ic, oc)
            w_base = kh * (KW * C_IN * C_OUT) + kw * (C_IN * C_OUT)  # using packed [KH,KW,C_IN,C_OUT]
            w_offs = w_base + offs_k[:, None] * C_OUT + offs_n[None, :]
            w_mask = mask_k[:, None] & mask_n[None, :]
            w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

            acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    # bias
    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b_vals[None, :]
    acc = acc - SUB

    # mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = acc * th

    # store NHWC: out[n, oh, ow, oc]
    out_offs = offs_m[:, None] * C_OUT + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offs, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-pack weight to layout [KH, KW, C_IN, C_OUT] contiguous
        with torch.no_grad():
            w = self.conv.weight.detach()  # [C_OUT, C_IN, KH, KW]
            w_packed = w.permute(2, 3, 1, 0).contiguous()  # [KH, KW, C_IN, C_OUT]
            self.register_buffer("w_packed", w_packed)
            self.register_buffer("bias_buf", self.conv.bias.detach().clone())

    def forward(self, x):
        x = x.contiguous()
        N, C_IN, H_IN, W_IN = x.shape
        KH = KW = self.kernel_size
        H_OUT = H_IN - KH + 1
        W_OUT = W_IN - KW + 1
        C_OUT = self.out_channels

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [N, H_IN, W_IN, C_IN]

        out_nhwc = torch.empty((N, H_OUT, W_OUT, C_OUT), device=x.device, dtype=x.dtype)

        SUB = float(self.subtract_value_1 + self.subtract_value_2)

        # Tile sizes
        BLOCK_M = 128
        BLOCK_N = 64
        # BLOCK_K must be >= 16 for tl.dot; pad C_IN up to next pow2 >= 16
        BLOCK_K = 16
        if C_IN > 16:
            # next power of two
            bk = 1
            while bk < C_IN:
                bk *= 2
            BLOCK_K = max(16, bk)

        M = N * H_OUT * W_OUT
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(C_OUT, BLOCK_N))

        conv2d_mish_nhwc_kernel[grid](
            x_nhwc, self.w_packed, self.bias_buf, out_nhwc,
            N, H_IN, W_IN,
            H_OUT, W_OUT,
            C_IN, C_OUT,
            KH, KW,
            SUB,
            BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Convert output back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out