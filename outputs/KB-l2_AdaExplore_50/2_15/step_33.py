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
    sum_ptr,      # [N*C] partial sums of x over spatial
    N, C, S,
    inv_S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    c = pid % C
    row_off = pid * S

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    s_x = tl.load(sum_ptr + pid)
    # mean of normed = scale * (s_x / S) + shift
    mean_normed = scale * (s_x * inv_S) + shift
    # final = scale * x + shift - mean_normed = scale * x + (shift - mean_normed)
    new_shift = shift - mean_normed

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_off + offs, mask=mask, other=0.0)
        out = scale * vals + new_shift
        tl.store(out_ptr + row_off + offs, out, mask=mask)


@triton.jit
def _per_nc_sum_kernel(
    x_ptr, sum_ptr,
    S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    row_off = pid * S
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        acc += vals
    s = tl.sum(acc, axis=0)
    tl.store(sum_ptr + pid, s)


def fused_bn_subtract_mean(x, gamma, beta, running_mean, running_var, eps):
    assert x.is_cuda and x.is_contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W

    scale_c = gamma / torch.sqrt(running_var + eps)
    shift_c = beta - scale_c * running_mean
    scale_c = scale_c.contiguous().to(x.dtype)
    shift_c = shift_c.contiguous().to(x.dtype)

    sum_nc = torch.empty((N * C,), device=x.device, dtype=torch.float32)

    BLOCK_S = 1024
    grid = (N * C,)
    _per_nc_sum_kernel[grid](x, sum_nc, S, BLOCK_S=BLOCK_S, num_warps=4, num_stages=2)

    out = torch.empty_like(x)
    inv_S = 1.0 / S
    _fused_bn_submean_kernel[grid](
        x, out, scale_c, shift_c, sum_nc,
        N, C, S, inv_S,
        BLOCK_S=BLOCK_S, num_warps=4, num_stages=2,
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
            x, bn.weight, bn.bias,
            bn.running_mean, bn.running_var,
            bn.eps,
        )