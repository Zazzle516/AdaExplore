import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_reduce_channels_last_kernel(
    x_ptr,           # [N, H, W, C] (channels_last view)
    out_ptr,         # [N, C]
    N, C, HW,
    scale,
    BLOCK_HW: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    # channels_last layout: offset = n*HW*C + hw*C + c
    base = n * HW * C + c

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    num_iters = (HW + BLOCK_HW - 1) // BLOCK_HW

    for i in range(0, num_iters):
        offs = i * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        v = tl.load(x_ptr + base + offs * C, mask=mask, other=0.0)
        acc += v

    s = tl.sum(acc, axis=0)
    tl.store(out_ptr + n * C + c, s * scale)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
    ],
    key=['SPATIAL'],
)
@triton.jit
def fused_scale_mean_kernel(
    x_ptr,
    out_ptr,
    SPATIAL,
    multiplier,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * SPATIAL

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    num_iters = (SPATIAL + BLOCK_SIZE - 1) // BLOCK_SIZE

    for i in range(0, num_iters):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < SPATIAL
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += v

    s = tl.sum(acc, axis=0)
    result = s * (multiplier / SPATIAL)
    tl.store(out_ptr + pid, result)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last)

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        y = self.conv_transpose(x)
        N, C, H, W = y.shape
        HW = H * W

        # Use the channels_last memory directly to avoid an extra .contiguous() copy.
        # In channels_last, memory layout is [N, H, W, C].
        if y.is_contiguous(memory_format=torch.channels_last):
            out = torch.empty((N, C, 1, 1), device=y.device, dtype=y.dtype)
            scale = float(self.multiplier) / float(HW)
            grid = (N * C,)
            # Choose BLOCK_HW as a power of two >= some reasonable tile.
            BLOCK_HW = 1024
            fused_reduce_channels_last_kernel[grid](
                y, out,
                N, C, HW,
                scale,
                BLOCK_HW=BLOCK_HW,
                num_warps=4,
                num_stages=2,
            )
            return out
        else:
            y = y.contiguous()
            out = torch.empty((N, C, 1, 1), device=y.device, dtype=y.dtype)
            grid = (N * C,)
            fused_scale_mean_kernel[grid](
                y, out,
                HW,
                float(self.multiplier),
            )
            return out