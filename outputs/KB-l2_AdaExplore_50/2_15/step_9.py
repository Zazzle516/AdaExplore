import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mean_sub_kernel(
    x_ptr,
    out_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    row_offset = pid * S

    # Compute mean
    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    mean = acc / S

    # Subtract mean
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row_offset + offs, vals - mean, mask=mask)


def mean_sub(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    # Choose BLOCK_S
    if S <= 1024:
        BLOCK_S = 1024
        num_warps = 4
    elif S <= 4096:
        BLOCK_S = 1024
        num_warps = 4
    else:
        BLOCK_S = 2048
        num_warps = 8
    _mean_sub_kernel[grid](x, out, N, C, S, BLOCK_S=BLOCK_S, num_warps=num_warps)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = x.contiguous()
        return mean_sub(x)