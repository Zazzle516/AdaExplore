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
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_epilogue_kernel[grid](
        gemm_out, out, n,
        SCALE=float(scaling_factor),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Use cuBLAS GEMM (highly tuned for 8192³ fp32 on Ada) via addmm,
        # then fuse sigmoid+scale+residual in a single elementwise kernel.
        gemm_out = torch.addmm(self.gemm.bias, x, self.gemm.weight.t())
        return fused_epilogue(gemm_out, self.scaling_factor)