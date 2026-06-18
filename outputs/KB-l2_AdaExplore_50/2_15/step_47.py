import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bn_submean_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C, S,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C
    row_start = pid * S

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    sum_val = tl.zeros((), dtype=tl.float32)
    for off in range(0, S, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        sum_val += tl.sum(tl.where(mask, x, 0.0), axis=0)
    # mean of (x*scale + shift) = scale*mean(x) + shift
    mean_x = sum_val / S
    mean_y = mean_x * scale + shift

    for off in range(0, S, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        y = x * scale + shift - mean_y
        tl.store(out_ptr + row_start + offs, y, mask=mask)


def fused_bn_submean(x, scale, shift):
    N, C, D, H, W = x.shape
    S = D * H * W
    x_flat = x.contiguous().view(N * C, S)
    out = torch.empty_like(x_flat)
    grid = (N * C,)
    fused_bn_submean_kernel[grid](
        x_flat, out, scale, shift,
        N, C, S,
        BLOCK_SIZE=2048, num_warps=8, num_stages=3,
    )
    return out.view(N, C, D, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        if self.training:
            x = self.batch_norm(x)
            mean = x.mean(dim=(2, 3, 4), keepdim=True)
            return x - mean
        else:
            bn = self.batch_norm
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            shift = bn.bias - scale * bn.running_mean
            scale = scale.contiguous()
            shift = shift.contiguous()
            return fused_bn_submean(x, scale, shift)