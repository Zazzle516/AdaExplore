import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 64, 'BLOCK_OC': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 64, 'BLOCK_OC': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32, 'BLOCK_OC': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32, 'BLOCK_OC': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32, 'BLOCK_OC': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 32, 'BLOCK_OC': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 32, 'BLOCK_OC': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 64, 'BLOCK_OC': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 64, 'BLOCK_OC': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 128, 'BLOCK_OC': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 64, 'BLOCK_OC': 4}, num_warps=8, num_stages=2),
    ],
    key=['IN_C', 'OUT_C', 'KH', 'KW', 'H_OUT', 'W_OUT'],
)
@triton.jit
def conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IN_C, H_IN, W_IN,
    OUT_C, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_on, stride_oh, stride_ow,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_OC: tl.constexpr,
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

    oc_inner = tl.arange(0, BLOCK_OC)

    # Loop over output channel chunks
    for oc_start in range(0, OUT_C, BLOCK_OC):
        oc_idx = oc_start + oc_inner
        oc_mask = oc_idx < OUT_C

        # Per-chunk accumulator: (BLOCK_OC, BLOCK_H, BLOCK_W) is too big.
        # Use separate (BLOCK_H, BLOCK_W) accumulators per oc by flattening into (BLOCK_H*BLOCK_W, BLOCK_OC).
        # Actually keep as 3D: (BLOCK_H, BLOCK_W, BLOCK_OC)
        acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_OC), dtype=tl.float32)

        # Loop over input channels and kernel positions
        for ic in range(0, IN_C):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    h_in = h_offs[:, None] + kh
                    w_in = w_offs[None, :] + kw
                    in_mask = hw_mask & (h_in < H_IN) & (w_in < W_IN)

                    x_ptrs = (x_ptr + pid_n * stride_xn + ic * stride_xc
                              + h_in * stride_xh + w_in * stride_xw)
                    x_val = tl.load(x_ptrs, mask=in_mask, other=0.0).to(tl.float32)

                    w_ptrs = (w_ptr + oc_idx * stride_wo + ic * stride_wi
                              + kh * stride_wh + kw * stride_ww)
                    w_val = tl.load(w_ptrs, mask=oc_mask, other=0.0).to(tl.float32)

                    # x_val: (BLOCK_H, BLOCK_W), w_val: (BLOCK_OC,)
                    # broadcast to (BLOCK_H, BLOCK_W, BLOCK_OC)
                    acc += x_val[:, :, None] * w_val[None, None, :]

        # Add bias
        bias = tl.load(b_ptr + oc_idx, mask=oc_mask, other=float('inf')).to(tl.float32)
        acc += bias[None, None, :]
        # Mask invalid oc entries to +inf
        acc = tl.where(oc_mask[None, None, :], acc, float('inf'))

        # Reduce min across BLOCK_OC dimension
        chunk_min = tl.min(acc, axis=2)
        min_acc = tl.minimum(min_acc, chunk_min)

    # tanh(tanh(x)) using 2/(1+exp(-2x)) - 1 (single exp per tanh)
    e1 = tl.exp(-2.0 * min_acc)
    t1 = 2.0 / (1.0 + e1) - 1.0
    e2 = tl.exp(-2.0 * t1)
    t2 = 2.0 / (1.0 + e2) - 1.0

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