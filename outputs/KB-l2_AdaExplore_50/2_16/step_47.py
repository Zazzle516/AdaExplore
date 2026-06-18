import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr, bias_ptr, out_ptr, n_elements, channel_stride, num_channels,
    add_value, scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # channel index from flat index: (offsets // channel_stride) % num_channels
    c_idx = (offsets // channel_stride) % num_channels
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x = x + b
    # Mish: x * tanh(softplus(x))
    x_safe = tl.where(x > 20.0, 0.0, x)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x_safe)))
    e2 = tl.exp(2.0 * sp)
    th = 1.0 - 2.0 / (e2 + 1.0)
    y = x * th
    y = y + add_value
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * scale
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_epilogue(x, bias, add_value, scale):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    # x is (N, C, H, W); channel stride = H*W
    N, C, H, W = x.shape
    channel_stride = H * W
    BLOCK_SIZE = 8192
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_epilogue_kernel[grid](
        x, bias, out, n, channel_stride, C,
        float(add_value), float(scale),
        BLOCK_SIZE=BLOCK_SIZE, num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        # Keep bias separate, run conv without bias and fuse bias into epilogue
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False)
        with torch.no_grad():
            self.conv_transpose.weight.copy_(conv.weight)
        self.bias = nn.Parameter(conv.bias.detach().clone())
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_epilogue(x, self.bias, self.add_value, self.scale)
        return x