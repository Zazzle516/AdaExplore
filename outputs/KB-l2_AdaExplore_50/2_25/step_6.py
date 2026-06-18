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
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    w_off = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)  # [BLOCK_W]

    h_mask = h_off < H_OUT
    w_mask = w_off < W_OUT
    sp_mask = h_mask[:, None] & w_mask[None, :]  # [BLOCK_H, BLOCK_W]

    oc_range = tl.arange(0, C_OUT_C)
    bias = tl.load(b_ptr + oc_range)  # [C_OUT_C]

    # Reshape spatial to a flat [BLOCK_H*BLOCK_W] dim and accumulate over channels
    SP: tl.constexpr = BLOCK_H * BLOCK_W
    acc = tl.zeros((SP, C_OUT_C), dtype=tl.float32)

    # flatten spatial coords
    oh_flat = tl.reshape(h_off[:, None] + tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32), (SP,))
    ow_flat = tl.reshape(tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32) + w_off[None, :], (SP,))
    sp_mask_flat = tl.reshape(sp_mask, (SP,))

    for ic in tl.static_range(0, C_IN_C):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh_flat + kh
                iw = ow_flat + kw
                in_off = ((pid_n * C_IN + ic) * H_IN + ih) * W_IN + iw
                x_val = tl.load(x_ptr + in_off, mask=sp_mask_flat, other=0.0)
                w_idx = (oc_range * C_IN + ic) * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val[:, None] * w_val[None, :]

    acc = acc + bias[None, :]
    min_val = tl.min(acc, axis=1)  # [SP]

    t1 = tl.extra.cuda.libdevice.tanh(min_val)
    t2 = tl.extra.cuda.libdevice.tanh(t1)

    # store
    out_h = tl.reshape(h_off[:, None] + tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32), (SP,))
    out_w = tl.reshape(tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32) + w_off[None, :], (SP,))
    out_off = (pid_n * H_OUT + out_h) * W_OUT + out_w
    tl.store(out_ptr + out_off, t2, mask=sp_mask_flat)


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

        BLOCK_H = 8
        BLOCK_W = 32
        grid = (N, triton.cdiv(H_OUT, BLOCK_H), triton.cdiv(W_OUT, BLOCK_W))

        conv_min_tanh_kernel[grid](
            x, w, b, out,
            N, C_IN, H_IN, W_IN,
            H_OUT, W_OUT,
            KH, KW,
            C_IN, C_OUT,
            BLOCK_H, BLOCK_W,
            num_warps=8,
            num_stages=2,
        )
        return out