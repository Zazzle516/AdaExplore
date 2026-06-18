import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _per_nc_reduce_kernel(
    x_ptr, sum_ptr, sumsq_ptr,
    S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    row_off = pid * S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK_S], dtype=tl.float32)
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        acc += vals
        acc_sq += vals * vals
    s = tl.sum(acc, axis=0)
    sq = tl.sum(acc_sq, axis=0)
    tl.store(sum_ptr + pid, s)
    tl.store(sumsq_ptr + pid, sq)


@triton.jit
def _fused_epilogue_kernel(
    x_ptr, out_ptr,
    sum_nc_ptr,    # [N, C] : per (n,c) sum of x over spatial
    scale_ptr,     # [C]    : gamma / sqrt(var_c + eps)
    N, C, S,
    inv_S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    n = pid // C
    c = pid % C
    row_off = pid * S

    s_nc = tl.load(sum_nc_ptr + pid)
    m_nc = s_nc * inv_S
    scale = tl.load(scale_ptr + c)

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_off + offs, mask=mask, other=0.0)
        out = scale * (vals - m_nc)
        tl.store(out_ptr + row_off + offs, out, mask=mask)


def fused_bn_subtract_mean(x, gamma, beta, running_mean, running_var,
                           training, momentum, eps):
    assert x.is_cuda and x.is_contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    M = N * S

    sum_nc = torch.empty((N, C), device=x.device, dtype=torch.float32)
    sumsq_nc = torch.empty((N, C), device=x.device, dtype=torch.float32)

    BLOCK_S = 1024
    grid = (N * C,)
    _per_nc_reduce_kernel[grid](x, sum_nc, sumsq_nc, S, BLOCK_S=BLOCK_S, num_warps=4)

    if training:
        # per-channel stats from sums
        sum_c = sum_nc.sum(dim=0)        # [C]
        sumsq_c = sumsq_nc.sum(dim=0)    # [C]
        mu_c = sum_c / M
        var_c = sumsq_c / M - mu_c * mu_c
        # update running stats (unbiased var for running_var)
        with torch.no_grad():
            if running_mean is not None:
                running_mean.mul_(1 - momentum).add_(mu_c, alpha=momentum)
            if running_var is not None and M > 1:
                var_unbiased = var_c * (M / (M - 1))
                running_var.mul_(1 - momentum).add_(var_unbiased, alpha=momentum)
        scale_c = gamma / torch.sqrt(var_c + eps)
    else:
        scale_c = gamma / torch.sqrt(running_var + eps)

    scale_c = scale_c.contiguous().to(x.dtype)
    out = torch.empty_like(x)
    inv_S = 1.0 / S
    _fused_epilogue_kernel[grid](
        x, out, sum_nc, scale_c, N, C, S, inv_S,
        BLOCK_S=BLOCK_S, num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        bn = self.batch_norm
        x = fused_bn_subtract_mean(
            x,
            bn.weight, bn.bias,
            bn.running_mean, bn.running_var,
            bn.training, bn.momentum, bn.eps,
        )
        return x