import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_H': 8, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_H': 4, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_H': 8, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_H': 4, 'BLOCK_W': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_H': 4, 'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_H': 4, 'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_H': 16, 'BLOCK_W': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_H': 8, 'BLOCK_W': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_H': 8, 'BLOCK_W': 64}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'H_OUT', 'W_OUT', 'KH', 'KW'],
)
@triton.jit
def conv2d_div_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, H_IN, W_IN,
    OC, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    inv_divisor, neg_slope,
    BLOCK_OC: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    num_w_tiles = tl.cdiv(W_OUT, BLOCK_W)
    pid_h = pid_hw // num_w_tiles
    pid_w = pid_hw % num_w_tiles

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    mask_oc = offs_oc < OC
    mask_h = offs_h < H_OUT
    mask_w = offs_w < W_OUT
    mask_hw = mask_h[:, None] & mask_w[None, :]  # [BLOCK_H, BLOCK_W]

    acc = tl.zeros((BLOCK_OC, BLOCK_H * BLOCK_W), dtype=tl.float32)

    x_batch_offset = pid_n * IC * H_IN * W_IN

    for ic in tl.static_range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                in_h = offs_h + kh  # [BLOCK_H]
                in_w = offs_w + kw  # [BLOCK_W]

                x_offset = (x_batch_offset
                            + ic * H_IN * W_IN
                            + in_h[:, None] * W_IN
                            + in_w[None, :])
                x_vals = tl.load(x_ptr + x_offset, mask=mask_hw, other=0.0)
                x_flat = tl.reshape(x_vals, (BLOCK_H * BLOCK_W,))

                w_offset = offs_oc * (IC * KH * KW) + ic * KH * KW + kh * KW + kw
                w_vals = tl.load(w_ptr + w_offset, mask=mask_oc, other=0.0)

                acc += w_vals[:, None] * x_flat[None, :]

    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b_vals[:, None]
    acc = acc * inv_divisor
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    HW = H_OUT * W_OUT
    out_h2 = offs_h[:, None] * W_OUT + offs_w[None, :]  # [BLOCK_H, BLOCK_W]
    out_hw_flat = tl.reshape(out_h2, (BLOCK_H * BLOCK_W,))
    mask_hw_flat = tl.reshape(mask_hw, (BLOCK_H * BLOCK_W,))

    out_offset = (pid_n * OC * HW
                  + offs_oc[:, None] * HW
                  + out_hw_flat[None, :])
    mask = mask_oc[:, None] & mask_hw_flat[None, :]
    tl.store(out_ptr + out_offset, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, H_IN, W_IN = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        H_OUT = H_IN - KH + 1
        W_OUT = W_IN - KW + 1

        out = torch.empty((N, OC, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        inv_divisor = 1.0 / float(self.divisor)
        neg_slope = 0.01

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(H_OUT, meta['BLOCK_H']) * triton.cdiv(W_OUT, meta['BLOCK_W']),
        )

        conv2d_div_lrelu_kernel[grid](
            x, w, b, out,
            N, IC, H_IN, W_IN,
            OC, H_OUT, W_OUT,
            KH, KW,
            inv_divisor, neg_slope,
        )
        return out