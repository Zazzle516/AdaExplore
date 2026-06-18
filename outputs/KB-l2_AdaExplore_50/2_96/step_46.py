import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_reduce_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    SCALE: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    MK: tl.constexpr,  # maxpool kernel
    BLOCK: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    PD = D // MK
    PH = H // MK
    PW = W // MK
    pooled_total = PD * PH * PW

    base = (n * C + c) * D * H * W

    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # iterate over pooled output positions in blocks
    for start in range(0, pooled_total, BLOCK):
        idx = start + offs
        mask = idx < pooled_total

        pd = idx // (PH * PW)
        rem = idx % (PH * PW)
        ph = rem // PW
        pw = rem % PW

        # max over MK^3 window
        max_val = tl.full([BLOCK], -float('inf'), dtype=tl.float32)
        for kd in tl.static_range(MK):
            for kh in tl.static_range(MK):
                for kw in tl.static_range(MK):
                    d = pd * MK + kd
                    h = ph * MK + kh
                    w = pw * MK + kw
                    off = base + d * H * W + h * W + w
                    v = tl.load(x_ptr + off, mask=mask, other=-float('inf'))
                    v = v * SCALE
                    max_val = tl.where(v > max_val, v, max_val)

        max_val = tl.where(mask, max_val, 0.0)
        acc = acc + max_val

    s = tl.sum(acc, axis=0)
    avg = s / pooled_total
    avg = tl.minimum(tl.maximum(avg, CLAMP_MIN), CLAMP_MAX)
    tl.store(out_ptr + pid, avg)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = float(scale)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        N, C, D, H, W = x.shape
        MK = self.maxpool_kernel_size
        # truncate to multiple of MK (matches MaxPool3d default behavior with no padding)
        D2 = (D // MK) * MK
        H2 = (H // MK) * MK
        W2 = (W // MK) * MK
        if D2 != D or H2 != H or W2 != W:
            x = x[:, :, :D2, :H2, :W2].contiguous()
            D, H, W = D2, H2, W2

        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
        grid = (N * C,)
        BLOCK = 128
        fused_reduce_kernel[grid](
            x, out,
            N, C, D, H, W,
            self.scale,
            self.clamp_min, self.clamp_max,
            MK,
            BLOCK,
            num_warps=4,
        )
        return out