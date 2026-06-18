import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_post_kernel(
    x_ptr,          # conv output: (N, OC, H, W)
    bias_ptr,       # (OC,)
    out_ptr,        # (N,)
    N, OC, HW,
    inv_hw,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    base = n * OC * HW

    for hw_start in range(0, HW, BLOCK_HW):
        hw_offs = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offs < HW
        ptrs = x_ptr + base + oc_offs[:, None] * HW + hw_offs[None, :]
        mask = oc_mask[:, None] & hw_mask[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=1)

    mean_vals = acc * inv_hw

    bias_vals = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    z = mean_vals + bias_vals

    neg_inf_val = float('-inf')
    z_masked = tl.where(oc_mask, z, neg_inf_val)
    max_z = tl.max(z_masked, axis=0)
    exp_z = tl.exp(z_masked - max_z)
    exp_z = tl.where(oc_mask, exp_z, 0.0)
    sum_exp = tl.sum(exp_z, axis=0)
    lse = max_z + tl.log(sum_exp)

    result = lse * 10.0
    tl.store(out_ptr + n, result)


def fused_post(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, OC, H, W = x.shape
    HW = H * W
    bias_flat = bias.reshape(-1).contiguous()
    out = torch.empty(N, device=x.device, dtype=torch.float32)

    BLOCK_OC = 1
    while BLOCK_OC < OC:
        BLOCK_OC *= 2

    grid = (N,)
    fused_post_kernel[grid](
        x, bias_flat, out,
        N, OC, HW,
        1.0 / HW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_HW=1024,
        num_warps=8,
        num_stages=3,
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