import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_tanh_pool_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    POH, POW,
    POOL: tl.constexpr,
    SUB1: tl.constexpr, SUB2: tl.constexpr,
    INV_POOL2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_nc = tl.program_id(1)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    P = POH * POW
    mask = offs < P

    poh = offs // POW
    pow_ = offs % POW

    base = pid_nc * H * W
    oh_start = poh * POOL
    ow_start = pow_ * POOL

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ph in tl.static_range(0, POOL):
        for pw in tl.static_range(0, POOL):
            ih = oh_start + ph
            iw = ow_start + pw
            in_off = base + ih * W + iw
            v = tl.load(x_ptr + in_off, mask=mask, other=0.0)
            v = v - SUB1
            e2 = tl.exp(2.0 * v)
            t = (e2 - 1.0) / (e2 + 1.0)
            v = t - SUB2
            acc += v

    acc = acc * INV_POOL2

    out_off = pid_nc * P + offs
    tl.store(out_ptr + out_off, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size_pool = int(kernel_size_pool)
        self.kernel_size = int(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        # use cuDNN conv
        y = F.conv2d(x, self.conv.weight, self.conv.bias)

        N, C, H, W = y.shape
        POOL = self.kernel_size_pool
        POH = H // POOL
        POW = W // POOL

        out = torch.empty((N, C, POH, POW), device=y.device, dtype=y.dtype)

        P = POH * POW
        BLOCK = 256
        grid = (triton.cdiv(P, BLOCK), N * C)

        fused_tanh_pool_kernel[grid](
            y, out,
            N, C, H, W,
            POH, POW,
            POOL,
            self.subtract1_value, self.subtract2_value,
            1.0 / (POOL * POOL),
            BLOCK,
            num_warps=4, num_stages=2,
        )
        return out