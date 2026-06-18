import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def mean_bias_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    OC, HW,
    inv_hw,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    oc = tl.program_id(1)
    
    base = n * OC * HW + oc * HW
    
    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offs = hw_start + tl.arange(0, BLOCK_HW)
        mask = hw_offs < HW
        vals = tl.load(x_ptr + base + hw_offs, mask=mask, other=0.0)
        acc += vals
    
    s = tl.sum(acc, axis=0)
    mean_val = s * inv_hw
    b = tl.load(bias_ptr + oc)
    z = mean_val + b
    
    tl.store(out_ptr + n * OC + oc, z)


@triton.jit
def lse_kernel(
    z_ptr,
    out_ptr,
    OC,
    BLOCK_OC: tl.constexpr,
):
    n = tl.program_id(0)
    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC
    
    z = tl.load(z_ptr + n * OC + oc_offs, mask=oc_mask, other=-float('inf'))
    max_z = tl.max(z, axis=0)
    exp_z = tl.exp(z - max_z)
    exp_z = tl.where(oc_mask, exp_z, 0.0)
    sum_exp = tl.sum(exp_z, axis=0)
    lse = max_z + tl.log(sum_exp)
    
    tl.store(out_ptr + n, lse * 10.0)


def fused_post(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, OC, H, W = x.shape
    HW = H * W
    bias_flat = bias.reshape(-1).contiguous()
    
    z = torch.empty((N, OC), device=x.device, dtype=torch.float32)
    out = torch.empty(N, device=x.device, dtype=torch.float32)
    
    grid1 = (N, OC)
    mean_bias_kernel[grid1](
        x, bias_flat, z,
        OC, HW,
        1.0 / HW,
        BLOCK_HW=4096,
        num_warps=8,
        num_stages=3,
    )
    
    BLOCK_OC = 1
    while BLOCK_OC < OC:
        BLOCK_OC *= 2
    
    grid2 = (N,)
    lse_kernel[grid2](
        z, out,
        OC,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
    )
    
    return out.view(N, 1)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        return fused_post(x, self.bias)