import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_sigmoid_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    s_blocks = tl.cdiv(S, BLOCK_S)
    n = pid // s_blocks
    sb = pid % s_blocks

    offs_c = tl.arange(0, BLOCK_C)
    offs_s = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_c = offs_c < C
    mask_s = offs_s < S

    base = n * C * S
    ptrs = x_ptr + base + offs_c[:, None] * S + offs_s[None, :]
    mask = mask_c[:, None] & mask_s[None, :]

    neg_inf = float('-inf')
    x = tl.load(ptrs, mask=mask, other=neg_inf)

    m = tl.max(x, axis=0)
    e = tl.exp(x - m[None, :])
    e = tl.where(mask_c[:, None], e, 0.0)
    s_sum = tl.sum(e, axis=0)
    sm = e / s_sum[None, :]
    out = 1.0 / (1.0 + tl.exp(-sm))

    out_ptrs = out_ptr + base + offs_c[:, None] * S + offs_s[None, :]
    tl.store(out_ptrs, out, mask=mask)


def fused_softmax_sigmoid(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    N, C = x.shape[0], x.shape[1]
    S = 1
    for d in x.shape[2:]:
        S *= d
    out = torch.empty_like(x)

    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    BLOCK_S = 128
    grid = (N * triton.cdiv(S, BLOCK_S),)
    fused_softmax_sigmoid_kernel[grid](
        x, out, N, C, S,
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=3,
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
        x = fused_softmax_sigmoid(x)
        return x