import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def sub_mean_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    row_start = pid * S

    # compute mean
    sum_val = tl.zeros((), dtype=tl.float32)
    for off in range(0, S, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
    mean = sum_val / S

    for off in range(0, S, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row_start + offs, x - mean, mask=mask)


def sub_mean(x):
    N, C, D, H, W = x.shape
    S = D * H * W
    x_flat = x.contiguous().view(N * C, S)
    out = torch.empty_like(x_flat)
    grid = (N * C,)
    BLOCK = 1024
    sub_mean_kernel[grid](x_flat, out, N, C, S, BLOCK_SIZE=BLOCK, num_warps=4)
    return out.view(N, C, D, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = sub_mean(x)
        return x