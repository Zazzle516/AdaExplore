import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mish_bn_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C, H, W,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * H * W
    mask = offs < total

    # decode channel
    hw = H * W
    chw = C * hw
    n_idx = offs // chw
    rem = offs - n_idx * chw
    c_idx = rem // hw

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # softplus
    sp = tl.log(1.0 + tl.exp(x))
    # tanh via sigmoid: tanh(z) = 2*sigmoid(2z) - 1
    th = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    y = x * th

    scale = tl.load(scale_ptr + c_idx, mask=mask, other=0.0)
    shift = tl.load(shift_ptr + c_idx, mask=mask, other=0.0)
    out = y * scale + shift

    tl.store(out_ptr + offs, out, mask=mask)


def mish_bn_apply(x, scale, shift):
    x = x.contiguous()
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    total = x.numel()
    BLOCK = 1024
    grid = ((total + BLOCK - 1) // BLOCK,)
    _mish_bn_kernel[grid](x, out, scale, shift, N, C, H, W, BLOCK=BLOCK)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps

    def forward(self, x):
        x = self.conv(x)
        if self.training:
            # need running stats updates; fall back
            y = torch.multiply(torch.tanh(F.softplus(x)), x)
            y = self.bn(y)
            return y
        else:
            # Compute BN scale/shift from running stats
            rm = self.bn.running_mean
            rv = self.bn.running_var
            w = self.bn.weight
            b = self.bn.bias
            invstd = torch.rsqrt(rv + self.eps)
            scale = (w * invstd).contiguous()
            shift = (b - rm * w * invstd).contiguous()
            return mish_bn_apply(x, scale, shift)