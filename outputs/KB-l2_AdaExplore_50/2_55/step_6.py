import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=4, num_stages=3),
    ],
    key=['N', 'KERNEL_SIZE'],
)
@triton.jit
def pool_sum_scale_kernel(
    in_ptr, out_ptr,
    M, N,
    stride_m,
    SCALE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per (row, N-tile). Each computes a partial sum of the
    # maxpool over its tile, then atomic-adds the scaled partial to out[row].
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # We load BLOCK_N elements, then reduce in pairs of KERNEL_SIZE via
    # max over the second axis after a reshape.
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    neg_inf = float('-inf')
    x = tl.load(in_ptr + pid_m * stride_m + offs_n, mask=mask_n, other=neg_inf)

    # Reshape to (BLOCK_N // KERNEL_SIZE, KERNEL_SIZE) and take max along last
    POOLED: tl.constexpr = BLOCK_N // KERNEL_SIZE
    x_r = tl.reshape(x, (POOLED, KERNEL_SIZE))
    pooled = tl.max(x_r, axis=1)

    # Replace -inf (fully masked windows at tail) with 0
    pooled = tl.where(pooled == neg_inf, 0.0, pooled)

    partial = tl.sum(pooled, axis=0) * SCALE
    tl.atomic_add(out_ptr + pid_m, partial)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        lin = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = nn.Parameter(lin.bias.detach().clone())

    def forward(self, x):
        x = x.contiguous().cuda()
        M = x.shape[0]
        N = self.out_features
        assert N % self.kernel_size == 0, "out_features must be divisible by kernel_size"

        # Use cuBLAS for the heavy matmul: y = x @ W^T + b
        y = torch.addmm(self.bias, x, self.weight.t())

        out = torch.zeros((M,), device=x.device, dtype=torch.float32)

        grid = lambda meta: (M, triton.cdiv(N, meta['BLOCK_N']))

        pool_sum_scale_kernel[grid](
            y, out,
            M, N,
            y.stride(0),
            SCALE=self.scale_factor,
            KERNEL_SIZE=self.kernel_size,
        )
        return out