import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    gemm_ptr, out_ptr,
    n_elements,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(gemm_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    out = sig * SCALE + x
    tl.store(out_ptr + offs, out, mask=mask)


def fused_epilogue(gemm_out, scaling_factor):
    n = gemm_out.numel()
    out = torch.empty_like(gemm_out)
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_epilogue_kernel[grid](
        gemm_out, out, n,
        SCALE=float(scaling_factor),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Use cuBLAS GEMM via F.linear, then fuse sigmoid+scale+residual.
        gemm_out = torch.nn.functional.linear(x, self.gemm.weight, self.gemm.bias)
        return fused_epilogue(gemm_out, self.scaling_factor)