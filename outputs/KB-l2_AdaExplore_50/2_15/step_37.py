import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bn_sub_mean_kernel(
    x_ptr,        # input from conv_transpose [N, C, D, H, W]
    out_ptr,      # output [N, C, D, H, W]
    scale_ptr,    # [C]: bn weight / sqrt(var+eps)
    bias_ptr,     # [C]: bn bias - mean * scale
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = (n * C + c) * S

    scale = tl.load(scale_ptr + c)
    bias = tl.load(bias_ptr + c)

    # First pass: compute mean of (x*scale+bias)
    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * scale + bias
        acc += tl.sum(tl.where(mask, y, 0.0), axis=0)
    mean = acc / S

    # Second pass: write (x*scale+bias) - mean
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * scale + bias - mean
        tl.store(out_ptr + base + offs, y, mask=mask)


def bn_sub_mean(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 5
    N, C, D, H, W = x.shape
    S = D * H * W
    x = x.contiguous()
    out = torch.empty_like(x)
    BLOCK_S = 1024
    grid = (N * C,)
    _bn_sub_mean_kernel[grid](
        x, out, scale, bias,
        N, C, S,
        BLOCK_S=BLOCK_S, num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)

        if self.training:
            x = self.batch_norm(x)
            x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)
            return x
        else:
            bn = self.batch_norm
            eps = bn.eps
            inv = torch.rsqrt(bn.running_var + eps)
            scale = bn.weight * inv
            bias = bn.bias - bn.running_mean * scale
            return bn_sub_mean(x, scale.contiguous(), bias.contiguous())