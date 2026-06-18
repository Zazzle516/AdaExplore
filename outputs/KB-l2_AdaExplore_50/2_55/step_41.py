import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 8192}, num_warps=16, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def pool_sum_scale_kernel(
    x_ptr, out_ptr,
    N,
    stride_xm,
    scale_factor: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per row m. Iterate over N in BLOCK_N chunks.
    pid_m = tl.program_id(0)
    row_ptr = x_ptr + pid_m * stride_xm

    acc = tl.zeros((1,), dtype=tl.float32)
    # number of pooled outputs per BLOCK_N chunk
    POOLED_BLOCK: tl.constexpr = BLOCK_N // KERNEL_SIZE

    total = tl.zeros((), dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        vals = tl.load(row_ptr + offs, mask=mask, other=float('-inf'))
        # reshape and max along kernel
        vals_2d = tl.reshape(vals, (POOLED_BLOCK, KERNEL_SIZE))
        pooled = tl.max(vals_2d, axis=1)
        total += tl.sum(pooled, axis=0)

    total = total * scale_factor
    tl.store(out_ptr + pid_m, total)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        # Use cuBLAS for the heavy matmul
        y = F.linear(x, self.matmul.weight, self.matmul.bias)
        M, N = y.shape

        out = torch.empty(M, device=x.device, dtype=torch.float32)

        grid = (M,)
        pool_sum_scale_kernel[grid](
            y, out,
            N,
            y.stride(0),
            scale_factor=self.scale_factor,
            KERNEL_SIZE=self.kernel_size,
        )
        return out