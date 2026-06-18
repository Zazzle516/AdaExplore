import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_norm_kernel(
    x_ptr, y_ptr, out_ptr,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * N
    y_row = y_ptr + row * N
    out_row = out_ptr + row * N

    # First pass: sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply epilogue
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_row + offs, mask=mask, other=0.0).to(tl.float32)
        x_norm = (x - mean) * rstd
        result = (x_norm + y) * y
        tl.store(out_row + offs, result, mask=mask)


def fused_norm_add_mul(x, y, eps):
    assert x.is_cuda and y.is_cuda
    x = x.contiguous()
    y = y.contiguous()
    batch_size, N = x.shape
    out = torch.empty_like(x)

    BLOCK_N = 2048
    grid = (batch_size,)
    fused_norm_kernel[grid](
        x, y, out, N, eps,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.eps = eps

    def forward(self, x, y):
        x = self.bmm(x)
        x = fused_norm_add_mul(x, y, self.eps)
        return x