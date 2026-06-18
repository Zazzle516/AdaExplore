import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, S,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    # offs index into (N, C, S) flattened
    c_idx = (offs // S) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # original_x = x
    # x1 = x + bias
    # x2 = x1 + original_x = x + bias + x
    # x3 = x2 * original_x = (2x + bias) * x
    # x4 = x3 + original_x = (2x + bias) * x + x
    res = (2.0 * x + b) * x + x

    tl.store(out_ptr + offs, res, mask=mask)


def fused_epilogue(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    total = x.numel()
    out = torch.empty_like(x)
    bias_flat = bias.contiguous().view(-1)
    assert bias_flat.numel() == C

    BLOCK_SIZE = 1024
    grid = lambda meta: ((total + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_epilogue_kernel[grid](
        x, bias_flat, out,
        N, C, S, total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        return fused_epilogue(x, self.bias)