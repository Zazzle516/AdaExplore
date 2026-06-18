import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_mish_implicit_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H_IN, W_IN,
    H_OUT, W_OUT,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    SUB: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_TOTAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    HW_OUT = H_OUT * W_OUT
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < HW_OUT
    mask_n = offs_n < C_OUT

    oh = offs_m // W_OUT
    ow = offs_m % W_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    # K dimension: KH*KW*C_IN. Decompose k as (kh*KW + kw)*C_IN + ic.
    KH_KW_CIN = KH * KW * C_IN
    # We have BLOCK_K == K_TOTAL padded to next power; ensure single iteration.
    mask_k = offs_k < KH_KW_CIN

    ic = offs_k % C_IN
    khkw = offs_k // C_IN
    kh = khkw // KW
    kw = khkw % KW

    x_stride_n = H_IN * W_IN * C_IN
    x_stride_h = W_IN * C_IN
    x_stride_w = C_IN

    # x is NHWC; batch base
    x_batch_base = pid_b * x_stride_n

    ih = oh[:, None] + kh[None, :]   # [BLOCK_M, BLOCK_K]
    iw = ow[:, None] + kw[None, :]
    x_offs = x_batch_base + ih * x_stride_h + iw * x_stride_w + ic[None, :]
    x_mask = mask_m[:, None] & mask_k[None, :]
    x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

    # w packed as [KH*KW*C_IN, C_OUT]
    w_offs = offs_k[:, None] * C_OUT + offs_n[None, :]
    w_mask = mask_k[:, None] & mask_n[None, :]
    w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

    acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b_vals[None, :]
    acc = acc - SUB

    # mish: x * tanh(softplus(x))  using stable softplus
    # softplus(x) = log1p(exp(-|x|)) + max(x,0)
    abs_a = tl.abs(acc)
    sp = tl.log(1.0 + tl.exp(-abs_a)) + tl.maximum(acc, 0.0)
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out_val = acc * th

    # store NHWC: out[n, oh, ow, oc]
    out_batch_base = pid_b * HW_OUT * C_OUT
    out_offs = out_batch_base + offs_m[:, None] * C_OUT + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offs, out_val, mask=out_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        with torch.no_grad():
            w = self.conv.weight.detach()  # [C_OUT, C_IN, KH, KW]
            # Pack as [KH, KW, C_IN, C_OUT] then view as [KH*KW*C_IN, C_OUT]
            w_packed = w.permute(2, 3, 1, 0).contiguous().view(
                kernel_size * kernel_size * in_channels, out_channels
            ).contiguous()
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

        K_TOTAL = KH * KW * C_IN
        BLOCK_K = _next_pow2(K_TOTAL)
        if BLOCK_K < 16:
            BLOCK_K = 16

        BLOCK_M = 128
        BLOCK_N = 64

        grid = (triton.cdiv(H_OUT * W_OUT, BLOCK_M), triton.cdiv(C_OUT, BLOCK_N), N)

        conv2d_mish_implicit_gemm_kernel[grid](
            x_nhwc, self.w_packed, self.bias_buf, out_nhwc,
            N, H_IN, W_IN,
            H_OUT, W_OUT,
            C_IN, C_OUT,
            KH, KW,
            SUB,
            BLOCK_M, BLOCK_N, BLOCK_K,
            K_TOTAL,
            num_warps=4, num_stages=3,
        )

        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out