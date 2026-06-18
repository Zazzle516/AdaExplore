import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
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
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr,
    K_TOTAL: tl.constexpr,
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

    # weight is shaped [OC, IC*KH*KW] (already contiguous packing)
    # input: [N, IC, H_IN, W_IN]
    x_base = pid_n * IC * H_IN * W_IN

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # Iterate over all K = IC*KH*KW positions, fully unrolled via static_range
    for k in tl.static_range(0, K_TOTAL):
        ic = k // (KH * KW)
        khkw = k % (KH * KW)
        kh = khkw // KW
        kw = khkw % KW

        in_h = out_h + kh
        in_w = out_w + kw

        x_offset = x_base + ic * H_IN * W_IN + in_h * W_IN + in_w
        x_vals = tl.load(x_ptr + x_offset, mask=mask_hw, other=0.0)

        w_offset = offs_oc * K_TOTAL + k
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
        # weight reshape to [OC, IC*KH*KW]
        w = self.conv.weight.contiguous().cuda()
        OC = w.shape[0]
        IC = w.shape[1]
        KH = w.shape[2]
        KW = w.shape[3]
        w_flat = w.view(OC, IC * KH * KW).contiguous()
        b = self.conv.bias.contiguous().cuda()

        N, _, H_IN, W_IN = x.shape
        H_OUT = H_IN - KH + 1
        W_OUT = W_IN - KW + 1

        out = torch.empty((N, OC, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        inv_divisor = 1.0 / float(self.divisor)
        neg_slope = 0.01

        K_TOTAL = IC * KH * KW

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(H_OUT * W_OUT, meta['BLOCK_HW']),
        )

        conv2d_div_lrelu_kernel[grid](
            x, w_flat, b, out,
            N, IC, H_IN, W_IN,
            OC, H_OUT, W_OUT,
            KH, KW,
            inv_divisor, neg_slope,
            K_TOTAL=K_TOTAL,
        )
        return out