import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_HW': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_HW': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'H_OUT', 'W_OUT', 'KH', 'KW'],
)
@triton.jit
def conv2d_div_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, H_IN, W_IN,
    OC, H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    K_PAD: tl.constexpr,
    neg_slope,
    BLOCK_OC: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_k = tl.arange(0, K_PAD)

    HW_OUT = H_OUT * W_OUT
    K_REAL = IC * KH * KW

    mask_oc = offs_oc < OC
    mask_hw = offs_hw < HW_OUT
    mask_k = offs_k < K_REAL

    out_h = offs_hw // W_OUT
    out_w = offs_hw % W_OUT

    # Decompose k into (ic, kh, kw)
    k_ic = offs_k // (KH * KW)
    k_rem = offs_k % (KH * KW)
    k_kh = k_rem // KW
    k_kw = k_rem % KW

    # X is NHWC: x[n, h, w, c] -> base + h*W_IN*IC + w*IC + c
    x_base = pid_n * H_IN * W_IN * IC

    # in_h[hw, k] = out_h[hw] + k_kh[k] ; in_w[hw,k] = out_w[hw] + k_kw[k]
    in_h = out_h[:, None] + k_kh[None, :]  # [BLOCK_HW, K_PAD]
    in_w = out_w[:, None] + k_kw[None, :]
    ic_idx = k_ic[None, :]  # broadcast

    x_offsets = x_base + in_h * (W_IN * IC) + in_w * IC + ic_idx
    x_mask = mask_hw[:, None] & mask_k[None, :]
    # x_tile : [BLOCK_HW, K_PAD]
    x_tile = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

    # W is laid out as [OC, K_REAL] contiguous
    w_offsets = offs_oc[:, None] * K_REAL + offs_k[None, :]
    w_mask = mask_oc[:, None] & mask_k[None, :]
    # w_tile : [BLOCK_OC, K_PAD]
    w_tile = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

    # acc = w_tile @ x_tile.T  -> [BLOCK_OC, BLOCK_HW]
    acc = tl.dot(w_tile, tl.trans(x_tile))

    b_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + b_vals[:, None]

    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    # Output also NHWC: out[n, h, w, c]
    out_base = pid_n * HW_OUT * OC
    out_offsets = out_base + offs_hw[None, :] * OC + offs_oc[:, None]
    mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        inv_divisor = 1.0 / float(divisor)
        # Pre-scale weight and bias by 1/divisor; reshape weight to [OC, IC*KH*KW]
        with torch.no_grad():
            w = self.conv.weight.detach().clone() * inv_divisor  # [OC, IC, KH, KW]
            b = self.conv.bias.detach().clone() * inv_divisor    # [OC]
            OC = w.shape[0]
            w_flat = w.reshape(OC, -1).contiguous()
        self.register_buffer('w_scaled', w_flat)
        self.register_buffer('b_scaled', b)

    def forward(self, x):
        x = x.cuda()
        N, IC, H_IN, W_IN = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        H_OUT = H_IN - KH + 1
        W_OUT = W_IN - KW + 1

        # Convert to NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        K_REAL = IC * KH * KW
        # Pad K to next power of 2, at least 16 for tl.dot
        K_PAD = 1
        while K_PAD < max(K_REAL, 16):
            K_PAD *= 2

        out_nhwc = torch.empty((N, H_OUT, W_OUT, OC), device=x.device, dtype=x.dtype)

        neg_slope = 0.01

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(H_OUT * W_OUT, meta['BLOCK_HW']),
        )

        conv2d_div_lrelu_kernel[grid](
            x_nhwc, self.w_scaled, self.b_scaled, out_nhwc,
            N, IC, H_IN, W_IN,
            OC, H_OUT, W_OUT,
            KH, KW,
            K_PAD,
            neg_slope,
        )
        # Convert back to NCHW
        return out_nhwc.permute(0, 3, 1, 2).contiguous()