import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
    ],
    key=['C_out', 'H_out', 'W_out', 'C_in'],
)
@triton.jit
def conv2d_hswish_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    C_IN: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oh = hw_offs // W_out
    ow = hw_offs % W_out

    oc_mask = oc_offs < C_out
    hw_mask = hw_offs < (H_out * W_out)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    x_base = pid_n * (C_in * H_in * W_in)
    hw_in_base = oh * W_in + ow  # [BLOCK_HW]

    KHKW = KH * KW
    CKHKW = C_IN * KHKW

    for ci in tl.static_range(C_IN):
        x_ci_base = x_base + ci * (H_in * W_in)
        w_ci_base = ci * KHKW
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                x_idx = x_ci_base + hw_in_base + (kh * W_in + kw)
                x_vals = tl.load(x_ptr + x_idx, mask=hw_mask, other=0.0)

                w_idx = oc_offs * CKHKW + w_ci_base + (kh * KW + kw)
                w_vals = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)

                acc += w_vals[:, None] * x_vals[None, :]

    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_vals[:, None]

    # fused hardswish + relu
    hs_mid = acc * (acc + 3.0) * (1.0 / 6.0)
    res = tl.where(acc <= 0.0, 0.0, tl.where(acc >= 3.0, acc, hs_mid))

    out_idx = pid_n * (C_out * H_out * W_out) + oc_offs[:, None] * (H_out * W_out) + hw_offs[None, :]
    mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_idx, res, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, C_in, H_in, W_in = x.shape
        C_out = w.shape[0]
        KH = w.shape[2]
        KW = w.shape[3]
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1

        out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

        grid = lambda META: (N, triton.cdiv(C_out, META['BLOCK_OC']), triton.cdiv(H_out * W_out, META['BLOCK_HW']))

        conv2d_hswish_relu_kernel[grid](
            x, w, b, out,
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            KH, KW,
            C_in,
        )
        return out