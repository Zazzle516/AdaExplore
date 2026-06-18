import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'BLOCK_K': 72}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'BLOCK_K': 72}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64, 'BLOCK_K': 72}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128, 'BLOCK_K': 72}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'H_OUT', 'W_OUT', 'KH', 'KW'],
)
@triton.jit
def conv2d_div_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H_IN, W_IN,
    OC, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    inv_divisor, neg_slope,
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_k = tl.arange(0, BLOCK_K)

    mask_oc = offs_oc < OC
    mask_hw = offs_hw < (H_OUT * W_OUT)

    out_h = offs_hw // W_OUT
    out_w = offs_hw % W_OUT

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    x_batch_ptr = x_ptr + pid_n * IC * H_IN * W_IN

    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K_TOTAL

        ic = k_idx // (KH * KW)
        rem = k_idx % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # weight: [OC, IC*KH*KW] -> offset = oc * K_TOTAL + k
        w_off = offs_oc[:, None] * K_TOTAL + k_idx[None, :]
        w_mask = mask_oc[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_OC, BLOCK_K]

        # input gather: [BLOCK_K, BLOCK_HW]
        in_h = out_h[None, :] + kh[:, None]
        in_w = out_w[None, :] + kw[:, None]
        x_off = ic[:, None] * (H_IN * W_IN) + in_h * W_IN + in_w
        x_mask = k_mask[:, None] & mask_hw[None, :]
        x_vals = tl.load(x_batch_ptr + x_off, mask=x_mask, other=0.0)

        acc += tl.dot(w_vals, x_vals)

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
        K_TOTAL = IC * KH * KW

        # Preshape weight to [OC, IC*KH*KW]
        w_flat = w.view(OC, K_TOTAL).contiguous()

        out = torch.empty((N, OC, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        inv_divisor = 1.0 / float(self.divisor)
        neg_slope = 0.01

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
            K_TOTAL,
            inv_divisor, neg_slope,
        )
        return out