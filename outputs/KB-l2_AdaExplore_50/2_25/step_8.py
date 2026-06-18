import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_min_tanh_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_IN, H_IN, W_IN,
    H_OUT, W_OUT,
    KH: tl.constexpr, KW: tl.constexpr,
    C_IN_C: tl.constexpr,
    C_OUT_C: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < (H_OUT * W_OUT)
    oh = sp_offs // W_OUT
    ow = sp_offs % W_OUT

    # Load all biases once: shape [C_OUT_C]
    oc_range = tl.arange(0, C_OUT_C)
    bias = tl.load(b_ptr + oc_range)  # [C_OUT_C]

    # Accumulator: [BLOCK_SP, C_OUT_C], initialized to bias
    acc = tl.zeros((BLOCK_SP, C_OUT_C), dtype=tl.float32) + bias[None, :]

    # Loop: ic, kh, kw  -> load x tile once, then FMA across all OC
    for ic in tl.static_range(0, C_IN_C):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh
                iw = ow + kw
                in_off = ((pid_n * C_IN + ic) * H_IN + ih) * W_IN + iw
                x_val = tl.load(x_ptr + in_off, mask=sp_mask, other=0.0)  # [BLOCK_SP]

                # weight slice for all OC for this (ic, kh, kw): shape [C_OUT_C]
                # w layout: [C_OUT, C_IN, KH, KW]
                w_off = (oc_range * C_IN + ic) * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_off)  # [C_OUT_C]

                acc += x_val[:, None] * w_val[None, :]

    # min across C_OUT axis
    min_val = tl.min(acc, axis=1)  # [BLOCK_SP]

    # tanh(tanh(.))
    t1 = tl.extra.cuda.libdevice.tanh(min_val)
    t2 = tl.extra.cuda.libdevice.tanh(t1)

    out_off = (pid_n * H_OUT + oh) * W_OUT + ow
    tl.store(out_ptr + out_off, t2, mask=sp_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()

        N, C_IN, H_IN, W_IN = x.shape
        C_OUT = self.out_channels
        KH = KW = self.kernel_size
        H_OUT = H_IN - KH + 1
        W_OUT = W_IN - KW + 1

        out = torch.empty((N, 1, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        BLOCK_SP = 128
        grid = (N, triton.cdiv(H_OUT * W_OUT, BLOCK_SP))

        conv_min_tanh_kernel[grid](
            x, w, b, out,
            N, C_IN, H_IN, W_IN,
            H_OUT, W_OUT,
            KH, KW,
            C_IN, C_OUT,
            BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )
        return out