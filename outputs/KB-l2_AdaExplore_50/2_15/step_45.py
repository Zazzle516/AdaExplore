import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_bn_submean_kernel(
    x_ptr, out_ptr,
    scale_ptr,    # [C]  gamma/sqrt(var+eps)
    shift_ptr,    # [C]  beta - scale*mean
    N, C, S,
    inv_S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    c = pid % C
    row_off = pid * S

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    # First pass: compute sum of x over spatial
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        acc += vals
    total = tl.sum(acc, axis=0)
    mean_x = total * inv_S
    # final mean of normed = scale * mean_x + shift
    # output = scale * x + shift - (scale * mean_x + shift) = scale * (x - mean_x)
    # so shift cancels out entirely! We don't even need it.

    # Second pass: store scale * (x - mean_x)
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        out = scale * (vals - mean_x)
        tl.store(out_ptr + row_off + offs, out, mask=mask)


def fused_bn_subtract_mean(x, gamma, running_var, eps):
    assert x.is_cuda and x.is_contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W

    scale_c = gamma / torch.sqrt(running_var + eps)
    scale_c = scale_c.contiguous().to(x.dtype)
    # shift not needed since subtracting mean cancels it
    shift_c = torch.empty_like(scale_c)

    out = torch.empty_like(x)
    inv_S = 1.0 / S
    BLOCK_S = 2048
    grid = (N * C,)
    _fused_bn_submean_kernel[grid](
        x, out, scale_c, shift_c,
        N, C, S, inv_S,
        BLOCK_S=BLOCK_S, num_warps=8, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        bn = self.batch_norm
        if bn.training:
            x = bn(x)
            x = x - x.mean(dim=(2, 3, 4), keepdim=True)
            return x
        return fused_bn_subtract_mean(
            x, bn.weight,
            bn.running_var,
            bn.eps,
        )