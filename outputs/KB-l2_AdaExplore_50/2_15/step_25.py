import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bn_sub_mean_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,  # per-channel scale (folded BN)
    shift_ptr,  # per-channel shift (folded BN)
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = (n * C + c) * S

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    # Pass 1: compute sum to get mean of (scale*x + shift)
    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * scale + shift
        acc += tl.sum(tl.where(mask, y, 0.0), axis=0)
    mean = acc / S

    # Pass 2: write (scale*x + shift) - mean
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * scale + shift - mean
        tl.store(out_ptr + base + offs, y, mask=mask)


def bn_sub_spatial_mean(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 5
    N, C, D, H, W = x.shape
    S = D * H * W
    x = x.contiguous()
    out = torch.empty_like(x)
    BLOCK_S = 1024
    grid = (N * C,)
    _bn_sub_mean_kernel[grid](x, out, scale, shift, N, C, S, BLOCK_S=BLOCK_S, num_warps=4)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fold BN affine at eval; for train we still need running update behavior.
        if not self.training:
            bn = self.batch_norm
            eps = bn.eps
            inv = torch.rsqrt(bn.running_var + eps)
            scale = bn.weight * inv
            shift = bn.bias - bn.running_mean * scale
            return bn_sub_spatial_mean(x, scale.contiguous(), shift.contiguous())
        else:
            x = self.batch_norm(x)
            x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)
            return x