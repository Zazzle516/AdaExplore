import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_add_hardswish_kernel(
    x_ptr, add_ptr, bias_ptr, out_ptr,
    n_elements, channel_stride, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    a = tl.load(add_ptr + offsets, mask=mask, other=0.0)
    
    # compute channel index for bias
    c = (offsets // channel_stride) % n_channels
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    v = x + a + b
    # v * hardswish(v) = v * v * relu6(v+3)/6
    hs = v * tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0) * (1.0 / 6.0)
    out = v * hs
    
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bias_add_hardswish(x, add_input, bias, n_channels, channel_stride):
    x = x.contiguous()
    add_input = add_input.contiguous()
    bias = bias.contiguous().view(-1)
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    BLOCK_SIZE = 4096
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_bias_add_hardswish_kernel[grid](
        x, add_input, bias, out,
        n_elements, channel_stride, n_channels,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.out_channels = out_channels
    
    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        # x shape: (N, C, D, H, W); channel stride = D*H*W
        N, C, D, H, W = x.shape
        channel_stride = D * H * W
        return fused_bias_add_hardswish(x, add_input, self.bias, C, channel_stride)