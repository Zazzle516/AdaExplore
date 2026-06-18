import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_lse_hswish_sub_clamp_kernel(
    x_ptr,      # input: [N, C, D, H, W]
    bias_ptr,   # scalar bias
    out_ptr,    # output: [N, 1, D, H, W]
    n_elements, # N*D*H*W
    C: tl.constexpr,
    spatial_stride,  # D*H*W
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # offsets index into [N, D*H*W] flattened
    # Map to input: for sample n, spatial idx s, input element at channel c is at
    # n * C * spatial_stride + c * spatial_stride + s
    n_idx = offsets // spatial_stride
    s_idx = offsets % spatial_stride
    base = n_idx * (C * spatial_stride) + s_idx

    # Compute max across channels
    max_val = tl.full([BLOCK_SIZE], -float('inf'), tl.float32)
    for c in tl.static_range(0, C):
        v = tl.load(x_ptr + base + c * spatial_stride, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, v)

    # Compute sum of exp
    sum_exp = tl.zeros([BLOCK_SIZE], tl.float32)
    for c in tl.static_range(0, C):
        v = tl.load(x_ptr + base + c * spatial_stride, mask=mask, other=0.0)
        sum_exp = sum_exp + tl.exp(v - max_val)

    lse = max_val + tl.log(sum_exp)

    # HardSwish: x * sigmoid(x+3) / 6
    hs = lse * tl.sigmoid(lse + 3.0) / 6.0

    # Subtract bias (scalar)
    b = tl.load(bias_ptr)
    out = hs - b

    # Clamp
    out = tl.minimum(tl.maximum(out, -1.0), 1.0)

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_lse_hswish_sub_clamp(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    N, C, D, H, W = x.shape
    spatial = D * H * W
    n_elements = N * spatial
    out = torch.empty((N, 1, D, H, W), device=x.device, dtype=x.dtype)
    bias_flat = bias.contiguous().view(-1)[:1]

    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_lse_hswish_sub_clamp_kernel[grid](
        x, bias_flat, out,
        n_elements, C, spatial,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_lse_hswish_sub_clamp(x, self.bias)
        return x