import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _epilogue_kernel(
    x_ptr, b_ptr, out_ptr,
    N, C, HW,
    inv_s,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    base = n * C * HW + c * HW
    bias_val = tl.load(b_ptr + c)
    
    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        v = v + bias_val
        # clamp(clamp(v,0,1)*s, 0, 1) / s
        # since we then divide by s, equivalent to clamp(clamp(v,0,1), 0, 1/s)
        v = tl.minimum(tl.maximum(v, 0.0), 1.0)
        v = v * (1.0 / inv_s)  # placeholder, we just keep it simple
        v = tl.minimum(tl.maximum(v, 0.0), 1.0)
        v = v * inv_s
        tl.store(out_ptr + base + idx, v, mask=mask)


@triton.jit
def _fused_epilogue(
    x_ptr, b_ptr, out_ptr,
    C, HW,
    scale, inv_scale,
    total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    
    # compute channel index: offs / HW % C
    c_idx = (offs // HW) % C
    
    v = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + c_idx, mask=mask, other=0.0)
    v = v + b
    v = tl.minimum(tl.maximum(v, 0.0), 1.0)
    v = v * scale
    v = tl.minimum(tl.maximum(v, 0.0), 1.0)
    v = v * inv_scale
    tl.store(out_ptr + offs, v, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = self.conv_transpose(x)
        out = torch.empty_like(x)
        N, C, H, W = x.shape
        HW = H * W
        total = N * C * HW
        bias_flat = self.bias.view(-1).contiguous()
        scale = float(self.scaling_factor)
        inv_scale = 1.0 / scale
        
        BLOCK = 1024
        grid = ((total + BLOCK - 1) // BLOCK,)
        _fused_epilogue[grid](
            x, bias_flat, out,
            C, HW,
            scale, inv_scale,
            total,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out