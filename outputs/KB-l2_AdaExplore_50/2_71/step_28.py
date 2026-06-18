import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 32}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'H_OUT', 'W_OUT', 'KH', 'KW'],
)
@triton.jit
def conv2d_div_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H_IN, W_IN,
    OC, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    inv_divisor, neg_slope,
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    mask_oc = offs_oc < OC
    mask_hw = offs_hw < (H_OUT * W_OUT)

    out_h = offs_hw // W_OUT
    out_w = offs_hw % W_OUT

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                in_h = out_h + kh
                in_w = out_w + kw

                # x: [N, IC, H_IN, W_IN]
                x_offset = (pid_n * IC * H_IN * W_IN
                            + ic * H_IN * W_IN
                            + in_h * W_IN + in_w)
                x_vals = tl.load(x_ptr + x_offset, mask=mask_hw, other=0.0)

                # w: [OC, IC, KH, KW]
                w_offset = offs_oc * (IC * KH * KW) + ic * KH * KW + kh * KW + kw
                w_vals = tl.load(w_ptr + w_offset, mask=mask_oc, other=0.0)

                acc += w_vals[:, None] * x_vals[None, :]

    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b_vals[:, None]
    acc = acc * inv_divisor

    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    out_offset = (pid_n * OC * H_OUT * W_OUT
                  + offs_oc[:, None] * (H_OUT * W_OUT)
                  + offs_hw[None, :])
    mask = mask_oc[:, None] & mask_hw[None, :]
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
            triton.cdiv(H_OUT * W_OUT, meta['BLOCK_HW']),
        )

        conv2d_div_lrelu_kernel[grid](
            x, w, b, out,
            N, IC, H_IN, W_IN,
            OC, H_OUT, W_OUT,
            KH, KW,
            inv_divisor, neg_slope,
        )
        return out