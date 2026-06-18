import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bias_tanh_kernel(
    out_ptr, sub_bias_ptr,
    total, OC, spatial,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    oc = (offs // spatial) % OC

    v = tl.load(out_ptr + offs, mask=mask, other=0.0)
    sb = tl.load(sub_bias_ptr + oc, mask=mask, other=0.0)
    z = v - sb
    # tanh via exp
    e2 = tl.exp(2.0 * z)
    res = (e2 - 1.0) / (e2 + 1.0)
    tl.store(out_ptr + offs, res, mask=mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        # Run the (heavy) transposed convolution via cuDNN
        out = F.conv_transpose2d(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        out = out.contiguous()

        N, OC, OH, OW = out.shape
        sub_bias = self.bias.view(-1).contiguous()

        total = N * OC * OH * OW
        spatial = OH * OW
        BLOCK = 1024
        grid2 = ((total + BLOCK - 1) // BLOCK,)
        _bias_tanh_kernel[grid2](
            out, sub_bias,
            total, OC, spatial,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out