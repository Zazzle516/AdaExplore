import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_lse_hswish_sub_clamp_kernel_cl(
    x_ptr,      # input: [N, D, H, W, C] (channels-last contiguous)
    out_ptr,    # output: [N, D, H, W] flattened
    bias_val,   # scalar bias passed as fp32 tensor scalar via constexpr workaround
    n_elements, # N*D*H*W
    C: tl.constexpr,
    BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # x is laid out so channels are contiguous: stride for s is C
    base = offsets * C  # [BLOCK]
    c_idx = tl.arange(0, C)  # [C]
    addrs = base[:, None] + c_idx[None, :]
    tile_mask = mask[:, None]

    vals = tl.load(x_ptr + addrs, mask=tile_mask, other=-float('inf'))

    max_val = tl.max(vals, axis=1)
    sum_exp = tl.sum(tl.exp(vals - max_val[:, None]), axis=1)
    lse = max_val + tl.log(sum_exp)

    hs = lse * tl.sigmoid(lse + 3.0) / 6.0
    out = hs - BIAS
    out = tl.minimum(tl.maximum(out, -1.0), 1.0)

    tl.store(out_ptr + offsets, out, mask=mask)


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

    n_idx = offsets // spatial_stride
    s_idx = offsets % spatial_stride
    base = n_idx * (C * spatial_stride) + s_idx

    c_idx = tl.arange(0, C)
    addrs = base[:, None] + c_idx[None, :] * spatial_stride
    tile_mask = mask[:, None]

    vals = tl.load(x_ptr + addrs, mask=tile_mask, other=-float('inf'))

    max_val = tl.max(vals, axis=1)
    sum_exp = tl.sum(tl.exp(vals - max_val[:, None]), axis=1)
    lse = max_val + tl.log(sum_exp)

    hs = lse * tl.sigmoid(lse + 3.0) / 6.0
    b = tl.load(bias_ptr)
    out = hs - b
    out = tl.minimum(tl.maximum(out, -1.0), 1.0)

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_lse_hswish_sub_clamp(x: torch.Tensor, bias_val: float) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float32
    N, C, D, H, W = x.shape
    spatial = D * H * W
    n_elements = N * spatial
    BLOCK_SIZE = 2048

    # Check if x is in channels_last_3d layout
    is_cl = x.is_contiguous(memory_format=torch.channels_last_3d)

    if is_cl:
        # Reinterpret as [N, D, H, W, C] contiguous
        # x has strides: (C*D*H*W, 1, C*H*W, C*W, C). Permute to NDHWC, then it's contiguous.
        x_ndhwc = x.permute(0, 2, 3, 4, 1).contiguous() if not is_cl else x
        # When is channels_last_3d, permuting (0,2,3,4,1) gives a contiguous tensor view
        x_view = x.permute(0, 2, 3, 4, 1)
        out = torch.empty((N, 1, D, H, W), device=x.device, dtype=x.dtype)
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        fused_lse_hswish_sub_clamp_kernel_cl[grid](
            x_view, out,
            0.0,  # unused
            n_elements, C, float(bias_val),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
        return out
    else:
        x = x.contiguous()
        out = torch.empty((N, 1, D, H, W), device=x.device, dtype=x.dtype)
        bias_t = torch.tensor([bias_val], device=x.device, dtype=torch.float32)
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        fused_lse_hswish_sub_clamp_kernel[grid](
            x, bias_t, out,
            n_elements, C, spatial,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
        return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        # Convert weights to channels_last_3d for potentially faster cuDNN algo
        self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)
        x = fused_lse_hswish_sub_clamp(x, self.bias.item())
        return x