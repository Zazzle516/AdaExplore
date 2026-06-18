import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_add_hardswish_kernel(
    x_ptr, add_ptr, bias_ptr, out_ptr,
    n_elements, channel_stride, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    a = tl.load(add_ptr + offsets, mask=mask, other=0.0)
    
    # bias broadcast: bias shape is (C, 1, 1, 1, 1), so per output element, the channel idx is
    # (offset // channel_stride) % n_channels
    c_idx = (offsets // channel_stride) % n_channels
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    v = x + a + b
    # hardswish(v) = v * relu6(v+3)/6
    hs_inner = v + 3.0
    hs_inner = tl.maximum(hs_inner, 0.0)
    hs_inner = tl.minimum(hs_inner, 6.0)
    hs = v * hs_inner / 6.0
    out = v * hs
    
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_add_hardswish(x, add_input, bias):
    x = x.contiguous()
    add_input = add_input.contiguous()
    bias_flat = bias.contiguous().view(-1)
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    # x shape: (N, C, D, H, W)
    N, C, D, H, W = x.shape
    channel_stride = D * H * W
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_add_hardswish_kernel[grid](
        x, add_input, bias_flat, out,
        n_elements, channel_stride, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        # Try channels_last_3d; if it ends up slower we'll skip in forward.
        self._use_channels_last = True
        try:
            self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last_3d)
        except Exception:
            self._use_channels_last = False
    
    def forward(self, x, add_input):
        if self._use_channels_last:
            try:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            except Exception:
                x = x.contiguous()
        else:
            x = x.contiguous()
        x = self.conv_transpose(x)
        if x.is_contiguous(memory_format=torch.channels_last_3d):
            add_input = add_input.contiguous(memory_format=torch.channels_last_3d)
        else:
            add_input = add_input.contiguous()
        out = _fused_no_bias(x, add_input)
        return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_hardswish_kernel(
    x_ptr, add_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    a = tl.load(add_ptr + offsets, mask=mask, other=0.0)
    
    v = x + a
    hs_inner = v + 3.0
    hs_inner = tl.maximum(hs_inner, 0.0)
    hs_inner = tl.minimum(hs_inner, 6.0)
    hs = v * hs_inner * (1.0 / 6.0)
    out = v * hs
    
    tl.store(out_ptr + offsets, out, mask=mask)


def _fused_no_bias(x, add_input):
    if not x.is_contiguous() and not x.is_contiguous(memory_format=torch.channels_last_3d):
        x = x.contiguous()
    if not add_input.is_contiguous() and not add_input.is_contiguous(memory_format=torch.channels_last_3d):
        add_input = add_input.contiguous()
    # Ensure same memory layout for both
    if x.is_contiguous(memory_format=torch.channels_last_3d):
        add_input = add_input.contiguous(memory_format=torch.channels_last_3d)
        out = torch.empty_like(x, memory_format=torch.channels_last_3d)
    else:
        add_input = add_input.contiguous()
        out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_hardswish_kernel[grid](
        x, add_input, out, n_elements,
    )
    return out