import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_sigmoid_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = pid_n * C * S
    # ptrs shape [BLOCK_C, BLOCK_S]
    ptrs = x_ptr + base + offs_c[:, None] * S + offs_s[None, :]
    mask = mask_c[:, None] & mask_s[None, :]

    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)  # [BLOCK_S]
    e = tl.exp(x - m[None, :])
    e = tl.where(mask, e, 0.0)
    z = tl.sum(e, axis=0)  # [BLOCK_S]
    sm = e / z[None, :]
    out = 1.0 / (1.0 + tl.exp(-sm))

    out_ptrs = out_ptr + base + offs_c[:, None] * S + offs_s[None, :]
    tl.store(out_ptrs, out, mask=mask)


def fused_softmax_sigmoid(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    N, C = x.shape[0], x.shape[1]
    S = 1
    for d in x.shape[2:]:
        S *= d
    x_c = x.contiguous()
    out = torch.empty_like(x_c)

    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_S = 128
    grid = (N, triton.cdiv(S, BLOCK_S))
    softmax_sigmoid_kernel[grid](
        x_c, out, N, C, S,
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        num_warps=4,
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

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_softmax_sigmoid(x)
        return x