import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


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

    # iterate over C_in, KH, KW
    for ci in tl.static_range(C_IN):
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                ih = oh + kh
                iw = ow + kw
                # input pointer: x[pid_n, ci, ih, iw]
                x_idx = pid_n * (C_in * H_in * W_in) + ci * (H_in * W_in) + ih * W_in + iw
                x_vals = tl.load(x_ptr + x_idx, mask=hw_mask, other=0.0)  # [BLOCK_HW]

                # weight: w[oc, ci, kh, kw]
                w_idx = oc_offs * (C_in * KH * KW) + ci * (KH * KW) + kh * KW + kw
                w_vals = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += w_vals[:, None] * x_vals[None, :]

    # add bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_vals[:, None]

    # hardswish: x * relu6(x+3)/6, then relu
    # combined: if x <= -3: 0; if x >= 3: x; else x*(x+3)/6, then relu (x>=0)
    # since hardswish(x) is nonneg only when x >= 0 (for x in [-3,0], hswish is negative; relu kills it)
    # For x >= 0: hswish(x) = x*min(x+3,6)/6 -> if x>=3: x, else x*(x+3)/6. Both >= 0.
    # So result = where(x<=0, 0, where(x>=3, x, x*(x+3)/6))
    zero = tl.zeros_like(acc)
    three = tl.full(acc.shape, 3.0, dtype=tl.float32)
    inv6 = 1.0 / 6.0
    hs = tl.where(acc >= 3.0, acc, acc * (acc + three) * inv6)
    res = tl.where(acc <= 0.0, zero, hs)

    # store
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

        BLOCK_OC = 32
        BLOCK_HW = 128

        grid = (N, triton.cdiv(C_out, BLOCK_OC), triton.cdiv(H_out * W_out, BLOCK_HW))

        conv2d_hswish_relu_kernel[grid](
            x, w, b, out,
            N, C_in, H_in, W_in,
            C_out, H_out, W_out,
            KH, KW,
            C_in,
            BLOCK_OC, BLOCK_HW,
            num_warps=4, num_stages=2,
        )
        return out