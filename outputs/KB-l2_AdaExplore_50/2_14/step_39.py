import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def row_sum_scale_kernel(
    mm_ptr, out_ptr,
    N,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    vals = tl.load(mm_ptr + pid * N + offs)
    s = tl.sum(vals, axis=0) * SCALE
    tl.store(out_ptr + pid, s)


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

        row_sum_scale_kernel[(M,)](
            mm, out_flat,
            N,
            SCALE=scale,
            BLOCK_N=N,
            num_warps=8,
            num_stages=2,
        )
        return out