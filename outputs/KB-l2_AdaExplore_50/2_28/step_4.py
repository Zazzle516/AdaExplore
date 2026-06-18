import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_norm_add_mul_kernel(
    x_ptr, y_ptr, out_ptr,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * N
    y_row = y_ptr + row * N
    o_row = out_ptr + row * N

    # Pass 1: compute mean and variance
    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, N, BLOCK_N):
        offs = off + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply epilogue
    for off in range(0, N, BLOCK_N):
        offs = off + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_row + offs, mask=mask, other=0.0).to(tl.float32)
        normed = (x - mean) * rstd
        out = (normed + y) * y
        tl.store(o_row + offs, out, mask=mask)


def fused_norm_add_mul(x, y, eps):
    assert x.is_cuda and y.is_cuda
    x = x.contiguous()
    y = y.contiguous()
    B, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = 2048
    grid = (B,)
    fused_norm_add_mul_kernel[grid](
        x, y, out, N, eps,
        BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.eps = eps

    def forward(self, x, y):
        x = self.bmm(x)
        return fused_norm_add_mul(x, y, self.eps)