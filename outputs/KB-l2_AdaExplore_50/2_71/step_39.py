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
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'OC', 'H_OUT', 'W_OUT', 'KH', 'KW'],
)
@triton.jit
def conv2d_div_lrelu_dot_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, H_IN, W_IN,
    OC, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    inv_divisor, neg_slope,
    K_TOTAL: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_k = tl.arange(0, K_TOTAL)

    mask_oc = offs_oc < OC
    mask_hw = offs_hw < (H_OUT * W_OUT)

    out_h = offs_hw // W_OUT
    out_w = offs_hw % W_OUT

    x_base = pid_n * IC * H_IN * W_IN

    # Compute im2col indices for the [BLOCK_HW, K_TOTAL] input patch
    # k = ic*KH*KW + kh*KW + kw
    ic_k = offs_k // (KH * KW)
    rem_k = offs_k % (KH * KW)
    kh_k = rem_k // KW
    kw_k = rem_k % KW

    # in_h: [BLOCK_HW, K_TOTAL], in_w: [BLOCK_HW, K_TOTAL]
    in_h = out_h[:, None] + kh_k[None, :]
    in_w = out_w[:, None] + kw_k[None, :]
    x_offsets = x_base + ic_k[None, :] * (H_IN * W_IN) + in_h * W_IN + in_w
    x_mask = mask_hw[:, None]
    x_tile = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)  # [BLOCK_HW, K_TOTAL]

    # Weight tile [K_TOTAL, BLOCK_OC]: w is [OC, K_TOTAL] -> transpose access
    w_offsets = offs_k[:, None] + offs_oc[None, :] * K_TOTAL
    w_mask = mask_oc[None, :]
    w_tile = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)  # [K_TOTAL, BLOCK_OC]

    # acc: [BLOCK_HW, BLOCK_OC]
    acc = tl.dot(x_tile, w_tile)

    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b_vals[None, :]
    acc = acc * inv_divisor
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    # Store transposed: out is [N, OC, H_OUT*W_OUT]
    out_offset = (pid_n * OC * H_OUT * W_OUT
                  + offs_oc[None, :] * (H_OUT * W_OUT)
                  + offs_hw[:, None])
    mask = mask_oc[None, :] & mask_hw[:, None]
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

        conv2d_div_lrelu_dot_kernel[grid](
            x, w_flat, b, out,
            N, IC, H_IN, W_IN,
            OC, H_OUT, W_OUT,
            KH, KW,
            inv_divisor, neg_slope,
            K_TOTAL=K_TOTAL,
        )
        return out