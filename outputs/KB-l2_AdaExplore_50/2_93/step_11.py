import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_OC': 128}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'OC', 'H_OUT', 'W_OUT'],
)
@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,           # [N, H_IN, W_IN, IC] NHWC
    w_ptr,           # [IC, KH, KW, OC] reshaped
    bias_ptr,        # [OC]
    out_ptr,         # [N, H_OUT, W_OUT, OC] NHWC
    N, IC, OC,
    H_IN, W_IN,
    H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    add_value: tl.constexpr,
    multiply_value: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)   # batch index
    pid_hw = tl.program_id(1)  # output spatial tile (over H_OUT*W_OUT)
    pid_oc = tl.program_id(2)  # output channel tile

    hw_start = pid_hw * BLOCK_N
    offs_hw = hw_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_hw = offs_hw < (H_OUT * W_OUT)

    h_out = offs_hw // W_OUT  # [BLOCK_N]
    w_out = offs_hw % W_OUT

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    mask_oc = offs_oc < OC

    acc = tl.zeros((BLOCK_N, BLOCK_OC), dtype=tl.float32)

    # For stride=2, kernel=4, pad=0: input coord = (out - k) / 2 if divisible
    # Loop over KH, KW (unrolled via constexpr)
    for kh in tl.static_range(0, KH):
        # h_in_num = h_out - kh
        h_in_num = h_out - kh
        h_valid = (h_in_num >= 0) & ((h_in_num % STRIDE) == 0)
        h_in = h_in_num // STRIDE
        h_valid = h_valid & (h_in < H_IN)

        for kw in tl.static_range(0, KW):
            w_in_num = w_out - kw
            w_valid = (w_in_num >= 0) & ((w_in_num % STRIDE) == 0)
            w_in = w_in_num // STRIDE
            w_valid = w_valid & (w_in < W_IN)

            valid = h_valid & w_valid & mask_hw  # [BLOCK_N]

            # x offsets: [N, H_IN, W_IN, IC]
            # x_ptr + pid_n * H_IN*W_IN*IC + h_in*W_IN*IC + w_in*IC + ic
            x_base = pid_n * (H_IN * W_IN * IC) + h_in * (W_IN * IC) + w_in * IC  # [BLOCK_N]

            # w offsets: [IC, KH, KW, OC]
            # w_ptr + ic*KH*KW*OC + kh*KW*OC + kw*OC + oc
            w_base = kh * (KW * OC) + kw * OC  # scalar

            # GEMM-K over IC
            # Process in chunks of BLOCK_IC
            BLOCK_IC: tl.constexpr = 16
            for ic_start in range(0, IC, BLOCK_IC):
                offs_ic = ic_start + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
                mask_ic = offs_ic < IC

                # Load x: [BLOCK_N, BLOCK_IC]
                x_ptrs = x_ptr + x_base[:, None] + offs_ic[None, :]
                x_mask = valid[:, None] & mask_ic[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                # Load w: [BLOCK_IC, BLOCK_OC]
                w_ptrs = w_ptr + offs_ic[:, None] * (KH * KW * OC) + w_base + offs_oc[None, :]
                w_mask = mask_ic[:, None] & mask_oc[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Add bias
    bias = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)  # [BLOCK_OC]
    acc = acc + bias[None, :] + add_value

    # min(v, 0)
    acc = tl.minimum(acc, 0.0)

    # GELU
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))
    out = gelu * multiply_value

    # Store: [N, H_OUT, W_OUT, OC]
    out_base = pid_n * (H_OUT * W_OUT * OC) + offs_hw[:, None] * OC + offs_oc[None, :]
    store_mask = mask_hw[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_base, out, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Pre-transpose weight to [IC, KH, KW, OC] for NHWC GEMM
        # Original weight: [IC, OC, KH, KW]
        with torch.no_grad():
            w = self.conv_transpose.weight.data
            w_nhwc = w.permute(0, 2, 3, 1).contiguous()  # [IC, KH, KW, OC]
        self.register_buffer('weight_nhwc', w_nhwc, persistent=False)
        self._cached_weight_version = self.conv_transpose.weight._version

    def _get_weight_nhwc(self):
        w = self.conv_transpose.weight
        if w._version != self._cached_weight_version or self.weight_nhwc.device != w.device:
            with torch.no_grad():
                w_nhwc = w.data.permute(0, 2, 3, 1).contiguous()
            self.weight_nhwc = w_nhwc.to(w.device)
            self._cached_weight_version = w._version
        return self.weight_nhwc

    def forward(self, x):
        # x: [N, IC, H_IN, W_IN]
        N, IC, H_IN, W_IN = x.shape
        OC = self.out_channels
        KH = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        KW = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[1]
        STRIDE = self.stride if isinstance(self.stride, int) else self.stride[0]

        H_OUT = (H_IN - 1) * STRIDE + KH
        W_OUT = (W_IN - 1) * STRIDE + KW

        # Convert input to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w_nhwc = self._get_weight_nhwc()
        bias = self.conv_transpose.bias

        out_nhwc = torch.empty((N, H_OUT, W_OUT, OC), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            N,
            triton.cdiv(H_OUT * W_OUT, meta['BLOCK_N']),
            triton.cdiv(OC, meta['BLOCK_OC']),
        )

        conv_transpose_fused_kernel[grid](
            x_nhwc, w_nhwc, bias, out_nhwc,
            N, IC, OC,
            H_IN, W_IN,
            H_OUT, W_OUT,
            KH, KW,
            STRIDE,
            self.add_value, self.multiply_value,
        )

        # Convert back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out