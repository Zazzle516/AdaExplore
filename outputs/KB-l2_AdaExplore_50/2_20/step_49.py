import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr, bias_ptr, out_ptr,
    C, S,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    c_idx = (offs // S) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    res = (2.0 * x + b) * x + x

    tl.store(out_ptr + offs, res, mask=mask)


def fused_epilogue(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x_c = x.contiguous()
    N, C, D, H, W = x_c.shape
    S = D * H * W
    total = x_c.numel()
    out = torch.empty_like(x_c)
    bias_flat = bias.contiguous().view(-1)

    BLOCK_SIZE = 2048
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_epilogue_kernel[grid](
        x_c, bias_flat, out,
        C, S, total,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
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
        # Use channels_last_3d for weights to allow cuDNN to pick a faster algorithm
        try:
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(
                memory_format=torch.channels_last_3d
            )
        except Exception:
            pass

    def forward(self, x):
        try:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        except Exception:
            x = x.contiguous()
        x = self.conv_transpose(x)
        return fused_epilogue(x, self.bias)