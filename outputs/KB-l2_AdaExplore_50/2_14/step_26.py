import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 4096}, num_warps=16, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def row_sum_scale_kernel(
    mm_ptr, out_ptr,
    M, N,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # one program per row; reduce N
    row = tl.program_id(0)
    base = row * N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        vals = tl.load(mm_ptr + base + offs, mask=mask, other=0.0)
        acc += vals
    s = tl.sum(acc, axis=0) * SCALE
    tl.store(out_ptr + row, s)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = float(scaling_factor)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.contiguous()
        w = self.weight
        # Run the heavy matmul via cuBLAS (fastest path on 4090)
        mm = torch.matmul(x, w.t())  # (M, N)
        M, N = mm.shape
        scale = self.scaling_factor * 0.5

        out = torch.empty((M, 1), device=x.device, dtype=mm.dtype)
        out_flat = out.view(M)

        grid = (M,)
        row_sum_scale_kernel[grid](
            mm, out_flat,
            M, N,
            SCALE=scale,
        )
        return out