import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _sub_mean_kernel(
    x_ptr,
    out_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = (n * C + c) * S

    # compute mean
    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    mean = acc / S

    # subtract mean
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x - mean
        tl.store(out_ptr + base + offs, y, mask=mask)


def sub_spatial_mean(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 5
    N, C, D, H, W = x.shape
    S = D * H * W
    x = x.contiguous()
    out = torch.empty_like(x)
    BLOCK_S = 1024
    grid = (N * C,)
    _sub_mean_kernel[grid](x, out, N, C, S, BLOCK_S=BLOCK_S, num_warps=4)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = sub_spatial_mean(x)
        return x