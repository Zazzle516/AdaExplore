import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mean_kernel(
    y_ptr,
    mean_ptr,
    HW,
    stride_n, stride_c,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    base = pid_n * stride_n + pid_c * stride_c
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    off = 0
    while off < HW:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        v = tl.load(y_ptr + base + idx, mask=mask, other=0.0)
        acc += v
        off += BLOCK
    total = tl.sum(acc, axis=0)
    tl.store(mean_ptr + pid_n * tl.num_programs(1) + pid_c, total / HW)


@triton.jit
def _finalize_kernel(
    mean_ptr,
    bias_ptr,
    out_ptr,
    OC,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_OC)
    mask_c = offs_c < OC

    mean = tl.load(mean_ptr + pid_n * OC + offs_c, mask=mask_c, other=0.0)
    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    z = mean + b
    z_masked = tl.where(mask_c, z, -float('inf'))
    mx = tl.max(z_masked, axis=0)
    e = tl.exp(z_masked - mx)
    e = tl.where(mask_c, e, 0.0)
    s_sum = tl.sum(e, axis=0)
    lse = mx + tl.log(s_sum)
    tl.store(out_ptr + pid_n, lse * 10.0)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.out_channels = out_channels

    def forward(self, x):
        y = self.conv_transpose(x)
        N, OC, H, W = y.shape
        HW = H * W
        y = y.contiguous()

        BLOCK = 8192
        means = torch.empty((N, OC), device=y.device, dtype=torch.float32)

        _mean_kernel[(N, OC)](
            y, means, HW,
            y.stride(0), y.stride(1),
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=3,
        )

        out = torch.empty((N,), device=y.device, dtype=torch.float32)
        bias_flat = self.bias.view(-1).contiguous()

        BLOCK_OC = 1
        while BLOCK_OC < OC:
            BLOCK_OC *= 2

        _finalize_kernel[(N,)](
            means, bias_flat, out,
            OC,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
        )

        return out.view(N, 1)