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

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    base = pid_n * C * S
    # ptrs[c, s] = base + c*S + s
    ptrs = base + c_offs[:, None] * S + s_offs[None, :]
    mask = c_mask[:, None] & s_mask[None, :]

    x = tl.load(x_ptr + ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)  # [BLOCK_S]
    e = tl.exp(x - m[None, :])
    e = tl.where(c_mask[:, None], e, 0.0)
    s_sum = tl.sum(e, axis=0)  # [BLOCK_S]
    sm = e / s_sum[None, :]
    out = 1.0 / (1.0 + tl.exp(-sm))
    tl.store(out_ptr + ptrs, out, mask=mask)


def softmax_sigmoid(x):
    N, C, D, H, W = x.shape
    S = D * H * W
    x = x.contiguous()
    out = torch.empty_like(x)
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2
    BLOCK_S = 64
    grid = (N, (S + BLOCK_S - 1) // BLOCK_S)
    softmax_sigmoid_kernel[grid](
        x, out, N, C, S,
        BLOCK_C=BLOCK_C, BLOCK_S=BLOCK_S,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        x = softmax_sigmoid(x)
        return x