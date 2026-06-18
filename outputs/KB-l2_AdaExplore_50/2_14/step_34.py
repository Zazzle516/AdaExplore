import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 8192, 'ROWS_PER_PROG': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 8192, 'ROWS_PER_PROG': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 8192, 'ROWS_PER_PROG': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 4096, 'ROWS_PER_PROG': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 4096, 'ROWS_PER_PROG': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 4096, 'ROWS_PER_PROG': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 2048, 'ROWS_PER_PROG': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 2048, 'ROWS_PER_PROG': 8}, num_warps=4, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def row_sum_scale_kernel(
    mm_ptr, out_ptr,
    M, N,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    offs = tl.arange(0, BLOCK_N)
    for i in tl.static_range(ROWS_PER_PROG):
        row = row_start + i
        row_valid = row < M
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for n_start in range(0, N, BLOCK_N):
            cur = n_start + offs
            mask = (cur < N) & row_valid
            vals = tl.load(mm_ptr + row * N + cur, mask=mask, other=0.0)
            acc += vals
        s = tl.sum(acc, axis=0) * SCALE
        tl.store(out_ptr + row, s, mask=row_valid)


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
        mm = torch.matmul(x, w.t())
        M, N = mm.shape
        scale = self.scaling_factor * 0.5

        out = torch.empty((M, 1), device=x.device, dtype=mm.dtype)
        out_flat = out.view(M)

        grid = lambda meta: (triton.cdiv(M, meta['ROWS_PER_PROG']),)
        row_sum_scale_kernel[grid](
            mm, out_flat,
            M, N,
            SCALE=scale,
        )
        return out