import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_sigmoid_kernel_tiled(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = pid_n * C * S
    ptrs = x_ptr + base + offs_c[:, None] * S + s_offs[None, :]
    full_mask = mask_c[:, None] & s_mask[None, :]

    x = tl.load(ptrs, mask=full_mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m[None, :])
    e = tl.where(full_mask, e, 0.0)
    z = tl.sum(e, axis=0)
    sm = e / z[None, :]
    out = 1.0 / (1.0 + tl.exp(-sm))

    out_ptrs = out_ptr + base + offs_c[:, None] * S + s_offs[None, :]
    tl.store(out_ptrs, out, mask=full_mask)


def fused_softmax_sigmoid(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    N, C = x.shape[0], x.shape[1]
    S = 1
    for d in x.shape[2:]:
        S *= d
    x_c = x.contiguous()
    out = torch.empty_like(x_c)

    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_S = 256
    grid = (N, triton.cdiv(S, BLOCK_S))
    softmax_sigmoid_kernel_tiled[grid](
        x_c, out, N, C, S,
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            output_padding=output_padding, bias=bias
        )

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = tuple(stride)
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = tuple(padding)
        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding, output_padding)
        else:
            self.output_padding = tuple(output_padding)

    def forward(self, x):
        x = x.contiguous()
        y = F.conv_transpose3d(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        y = fused_softmax_sigmoid(y)
        return y