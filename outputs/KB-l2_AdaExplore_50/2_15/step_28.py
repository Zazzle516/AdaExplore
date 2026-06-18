import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bn_sub_mean_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,  # per-channel scale (after BN folding)
    shift_ptr,  # per-channel shift (after BN folding)
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = (n * C + c) * S

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    # First pass: compute sum of x; mean of (x*scale + shift) = scale*mean(x) + shift
    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    mean_x = acc / S
    mean_y = mean_x * scale + shift

    # Second pass: write y - mean
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * scale + (shift - mean_y)
        tl.store(out_ptr + base + offs, y, mask=mask)


def bn_sub_mean(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 5
    N, C, D, H, W = x.shape
    S = D * H * W
    x = x.contiguous()
    out = torch.empty_like(x)
    BLOCK_S = 2048
    grid = (N * C,)
    _bn_sub_mean_kernel[grid](x, out, scale, shift, N, C, S, BLOCK_S=BLOCK_S, num_warps=8, num_stages=2)
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
        if self.training:
            x = self.batch_norm(x)
            x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)
            return x
        else:
            # Fold BN affine using running stats
            rm = self.batch_norm.running_mean
            rv = self.batch_norm.running_var
            eps = self.batch_norm.eps
            w = self.batch_norm.weight
            b = self.batch_norm.bias
            invstd = torch.rsqrt(rv + eps)
            scale = (w * invstd).contiguous()
            shift = (b - rm * scale).contiguous()
            return bn_sub_mean(x, scale, shift)