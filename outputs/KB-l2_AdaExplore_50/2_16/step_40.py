import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cudnn.benchmark = True


@triton.jit
def fused_epilogue_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, channel_stride, num_channels,
    ADD_VALUE: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # bias indexing: NCHW layout, channel = (offset // channel_stride) % num_channels
    c = (offsets // channel_stride) % num_channels
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    x = x + b
    # Mish: x * tanh(softplus(x))
    x_safe = tl.where(x > 20.0, 0.0, x)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x_safe)))
    e2 = tl.exp(2.0 * sp)
    th = 1.0 - 2.0 / (e2 + 1.0)
    y = x * th + ADD_VALUE
    y = tl.minimum(tl.maximum(y, -1.0), 1.0) * SCALE
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_epilogue(x, bias, add_value, scale):
    x = x.contiguous()
    n = x.numel()
    # x shape: (N, C, H, W)
    C = x.shape[1]
    channel_stride = x.shape[2] * x.shape[3]
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_epilogue_kernel[grid](
        x, bias, x, n, channel_stride, C,
        ADD_VALUE=float(add_value), SCALE=float(scale),
        BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2,
    )
    return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        # Original conv with bias to preserve parameter shapes
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        # Run conv without bias, then fuse bias add into epilogue
        w = self.conv_transpose.weight
        b = self.conv_transpose.bias
        x = F.conv_transpose2d(
            x, w, None,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
            groups=self.conv_transpose.groups,
            dilation=self.conv_transpose.dilation,
        )
        x = fused_epilogue(x, b, self.add_value, self.scale)
        return x