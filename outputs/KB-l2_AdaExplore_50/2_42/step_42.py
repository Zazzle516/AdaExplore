import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mean_partial_kernel(
    y_ptr,
    partial_ptr,
    HW,
    stride_n, stride_c,
    OC,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_s = tl.program_id(2)
    base = pid_n * stride_n + pid_c * stride_c
    chunk = (HW + SPLIT - 1) // SPLIT
    start = pid_s * chunk
    end = start + chunk
    if end > HW:
        end = HW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    off = start
    while off < end:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < end
        v = tl.load(y_ptr + base + idx, mask=mask, other=0.0)
        acc += v
        off += BLOCK
    total = tl.sum(acc, axis=0)
    tl.store(partial_ptr + (pid_n * OC + pid_c) * SPLIT + pid_s, total)


@triton.jit
def _finalize_kernel(
    partial_ptr,
    bias_ptr,
    out_ptr,
    HW,
    OC,
    SPLIT: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_OC)
    mask_c = offs_c < OC

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    for s in range(0, SPLIT):
        v = tl.load(partial_ptr + (pid_n * OC + offs_c) * SPLIT + s, mask=mask_c, other=0.0)
        acc += v
    mean = acc / HW
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

        SPLIT = 32
        BLOCK = 4096
        partial = torch.empty((N, OC, SPLIT), device=y.device, dtype=torch.float32)

        _mean_partial_kernel[(N, OC, SPLIT)](
            y, partial, HW,
            y.stride(0), y.stride(1),
            OC,
            SPLIT=SPLIT,
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=4,
        )

        out = torch.empty((N,), device=y.device, dtype=torch.float32)
        bias_flat = self.bias.view(-1).contiguous()

        BLOCK_OC = 1
        while BLOCK_OC < OC:
            BLOCK_OC *= 2

        _finalize_kernel[(N,)](
            partial, bias_flat, out,
            HW, OC,
            SPLIT=SPLIT,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
        )

        return out.view(N, 1)