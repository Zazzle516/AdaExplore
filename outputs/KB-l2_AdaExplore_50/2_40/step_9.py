import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_epilogue_kernel(
    x_ptr, out_ptr, n_elements,
    SCALE_PLUS_ONE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * SCALE_PLUS_ONE, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous()
        # Use cuBLAS via torch.addmm for the GEMM + bias
        # y = bias + x @ weight.T
        y = torch.addmm(self.matmul.bias, x, self.matmul.weight.t())
        # Fused scale epilogue: y * (scale + 1)
        scale_plus_one = self.scaling_factor + 1.0
        out = torch.empty_like(y)
        n_elements = y.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        scale_epilogue_kernel[grid](
            y, out, n_elements,
            SCALE_PLUS_ONE=scale_plus_one,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        return out