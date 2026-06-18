import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 4096}, num_warps=16, num_stages=2),
    ],
    key=['OUT_FEAT'],
)
@triton.jit
def pool_sum_scale_kernel(
    inp_ptr, out_ptr,
    OUT_FEAT: tl.constexpr,
    POOLED: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per batch row. Reduces over OUT_FEAT in BLOCK_N chunks.
    pid = tl.program_id(0)
    row_ptr = inp_ptr + pid * OUT_FEAT

    acc = tl.zeros((), dtype=tl.float32)

    # BLOCK_N must be a multiple of KERNEL_SIZE
    for n_start in range(0, OUT_FEAT, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < OUT_FEAT
        vals = tl.load(row_ptr + offs, mask=mask, other=-float('inf'))
        # Reshape and maxpool along last dim
        reshaped = tl.reshape(vals, (BLOCK_N // KERNEL_SIZE, KERNEL_SIZE))
        pooled = tl.max(reshaped, axis=1)
        acc += tl.sum(pooled, axis=0)

    acc = acc * SCALE
    tl.store(out_ptr + pid, acc)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        # Linear layer for parameters
        self.matmul = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        # Use cuBLAS for the heavy GEMM (linear). This still runs the heavy op at runtime.
        # y = x @ W^T + b   shape [B, OUT_FEAT]
        y = torch.addmm(self.matmul.bias, x, self.matmul.weight.t())

        batch_size = x.shape[0]
        out = torch.empty(batch_size, device=x.device, dtype=x.dtype)

        pooled_len = self.out_features // self.kernel_size

        grid = (batch_size,)
        pool_sum_scale_kernel[grid](
            y, out,
            self.out_features,
            pooled_len,
            self.kernel_size,
            self.scale_factor,
        )
        return out