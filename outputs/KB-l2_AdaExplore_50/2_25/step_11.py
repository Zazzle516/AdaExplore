import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 64}, num_warps=4, num_stages=3),
    ],
    key=['IN_C', 'OUT_C', 'KH', 'KW', 'H_OUT', 'W_OUT'],
)
@triton.jit
def conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IN_C: tl.constexpr, H_IN, W_IN,
    OUT_C: tl.constexpr, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_on, stride_oh, stride_ow,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    num_w_blocks = tl.cdiv(W_OUT, BLOCK_W)
    pid_h = pid_hw // num_w_blocks
    pid_w = pid_hw % num_w_blocks

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    h_mask = h_offs < H_OUT
    w_mask = w_offs < W_OUT
    hw_mask = h_mask[:, None] & w_mask[None, :]

    # Initialize min accumulator with +inf
    min_acc = tl.full((BLOCK_H, BLOCK_W), float('inf'), dtype=tl.float32)

    x_base = x_ptr + pid_n * stride_xn

    # Loop over output channels
    for oc in range(0, OUT_C):
        bias = tl.load(b_ptr + oc).to(tl.float32)
        acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

        # Loop over kernel positions (static) then input channels (dynamic)
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                h_in = h_offs[:, None] + kh
                w_in = w_offs[None, :] + kw
                in_mask = hw_mask & (h_in < H_IN) & (w_in < W_IN)
                spatial_off = h_in * stride_xh + w_in * stride_xw
                for ic in range(0, IN_C):
                    x_ptrs = x_base + ic * stride_xc + spatial_off
                    x_val = tl.load(x_ptrs, mask=in_mask, other=0.0)

                    w_val = tl.load(w_ptr + oc * stride_wo + ic * stride_wi
                                    + kh * stride_wh + kw * stride_ww)
                    acc += x_val * w_val

        acc += bias
        min_acc = tl.minimum(min_acc, acc)

    # tanh(tanh(x)) - use libdevice
    t1 = (2.0 / (1.0 + tl.exp(-2.0 * min_acc))) - 1.0
    t2 = (2.0 / (1.0 + tl.exp(-2.0 * t1))) - 1.0

    out_ptrs = (out_ptr + pid_n * stride_on
                + h_offs[:, None] * stride_oh + w_offs[None, :] * stride_ow)
    tl.store(out_ptrs, t2, mask=hw_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv.weight.contiguous().cuda()
        bias = self.conv.bias.contiguous().cuda()

        N, IN_C, H_IN, W_IN = x.shape
        OUT_C = weight.shape[0]
        KH = weight.shape[2]
        KW = weight.shape[3]
        H_OUT = H_IN - KH + 1
        W_OUT = W_IN - KW + 1

        out = torch.empty((N, 1, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        grid = lambda meta: (
            N,
            triton.cdiv(H_OUT, meta['BLOCK_H']) * triton.cdiv(W_OUT, meta['BLOCK_W']),
        )

        conv_min_tanh_kernel[grid](
            x, weight, bias, out,
            N, IN_C, H_IN, W_IN,
            OUT_C, H_OUT, W_OUT,
            KH, KW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
            out.stride(0), out.stride(2), out.stride(3),
        )
        return out