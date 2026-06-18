import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel_v2(
    x_ptr, bias_ptr, out_ptr,
    n_elements, C, DHW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c_idx = (offs // DHW) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    out = (2.0 * x + b) * x + x
    tl.store(out_ptr + offs, out, mask=mask)


def fused_epilogue(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    bias_flat = bias.contiguous().view(-1)
    N, C, D, H, W = x.shape
    DHW = D * H * W
    n_elements = x.numel()
    BLOCK_SIZE = 4096
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    # In-place: write back into x to halve memory traffic
    fused_epilogue_kernel_v2[grid](
        x, bias_flat, x, n_elements, C, DHW, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4, num_stages=2,
    )
    return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        return fused_epilogue(x, self.bias)