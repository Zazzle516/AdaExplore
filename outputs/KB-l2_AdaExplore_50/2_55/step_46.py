import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=8, num_stages=3),
    ],
    key=['IN_FEAT', 'OUT_FEAT'],
)
@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    IN_FEAT: tl.constexpr,
    OUT_FEAT: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program per batch row.
    # Computes: linear -> maxpool1d(kernel_size) -> sum -> scale
    # Strategy: iterate over output_features in tiles of (KERNEL_SIZE * something),
    # compute matmul column-by-column tile, take max over kernel_size groups,
    # accumulate the sum.
    pid = tl.program_id(0)

    # Load x row once into BLOCK_K-sized chunks; we'll re-load per tile.
    # We process output features in chunks. For each chunk of BLOCK_N output features,
    # compute dot products, then reduce.

    # We'll use BLOCK_N = 64 output features per inner iteration.
    BLOCK_N: tl.constexpr = 64

    POOLED: tl.constexpr = OUT_FEAT // KERNEL_SIZE

    acc_sum = tl.zeros((1,), dtype=tl.float32)
    total = tl.zeros((), dtype=tl.float32)

    # Pointer to x row
    x_row_ptr = x_ptr + pid * IN_FEAT

    # Iterate over output feature tiles
    n_tiles = OUT_FEAT // BLOCK_N

    for n_idx in range(0, n_tiles):
        n_start = n_idx * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]

        # Compute matmul: out[n] = sum_k x[k] * w[n, k] + b[n]
        # for n in offs_n
        accum = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(0, IN_FEAT, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < IN_FEAT
            x_vals = tl.load(x_row_ptr + offs_k, mask=k_mask, other=0.0)  # [BLOCK_K]
            # w shape: [OUT_FEAT, IN_FEAT]; row-major
            w_ptrs = w_ptr + offs_n[:, None] * IN_FEAT + offs_k[None, :]
            w_vals = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)  # [BLOCK_N, BLOCK_K]
            accum += tl.sum(w_vals * x_vals[None, :], axis=1)

        # Add bias
        b_vals = tl.load(b_ptr + offs_n)
        accum += b_vals  # [BLOCK_N]

        # Now do maxpool over KERNEL_SIZE groups
        # Reshape accum [BLOCK_N] -> [BLOCK_N // KERNEL_SIZE, KERNEL_SIZE]
        # then max over last dim, then sum
        reshaped = tl.reshape(accum, (BLOCK_N // KERNEL_SIZE, KERNEL_SIZE))
        pooled = tl.max(reshaped, axis=1)  # [BLOCK_N // KERNEL_SIZE]
        total += tl.sum(pooled, axis=0)

    total = total * SCALE
    tl.store(out_ptr + pid, total)


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
        w = self.matmul.weight.contiguous()
        b = self.matmul.bias.contiguous()
        batch_size = x.shape[0]
        out = torch.empty(batch_size, device=x.device, dtype=x.dtype)

        grid = (batch_size,)
        fused_kernel[grid](
            x, w, b, out,
            self.in_features,
            self.out_features,
            self.kernel_size,
            self.scale_factor,
        )
        return out