import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mean_reduce_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    # one program per batch
    n = tl.program_id(0)
    total = C * S
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    
    # iterate over all elements
    num_iters = (total + BLOCK_S - 1) // BLOCK_S
    for i in range(num_iters):
        offs = i * BLOCK_S + tl.arange(0, BLOCK_S)
        mask = offs < total
        vals = tl.load(x_ptr + n * total + offs, mask=mask, other=0.0)
        acc += vals
    
    s = tl.sum(acc, axis=0)
    s = s / total
    tl.store(out_ptr + n, s)


def mean_reduce(x):
    N, C, D, H, W = x.shape
    S = D * H * W
    out = torch.empty(N, device=x.device, dtype=x.dtype)
    BLOCK_S = 1024
    mean_reduce_kernel[(N,)](x, out, N, C, S, BLOCK_S=BLOCK_S, num_warps=8)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = mean_reduce(x)
        return x