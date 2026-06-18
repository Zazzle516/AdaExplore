import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Phase 1: compute per-(N,C) spatial sum and sum-of-squares over D*H*W
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
    ],
    key=['S'],
)
@triton.jit
def _reduce_nc_kernel(
    x_ptr,           # [N, C, S]
    sum_ptr,         # [N, C]
    sumsq_ptr,       # [N, C]
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * S
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.where(mask, x, 0.0)
        acc2 += tl.where(mask, x * x, 0.0)
    s = tl.sum(acc, axis=0)
    ss = tl.sum(acc2, axis=0)
    tl.store(sum_ptr + pid, s)
    tl.store(sumsq_ptr + pid, ss)


# Phase 2: out[n,c,s] = (x[n,c,s] - sp_mean[n,c]) * scale[c]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=3),
    ],
    key=['S'],
)
@triton.jit
def _apply_kernel(
    x_ptr,           # [N, C, S]
    out_ptr,         # [N, C, S]
    sp_mean_ptr,     # [N, C]
    scale_ptr,       # [C]
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C
    base = pid * S
    sp_mean = tl.load(sp_mean_ptr + pid)
    scale = tl.load(scale_ptr + c)
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = (x - sp_mean) * scale
        tl.store(out_ptr + base + offs, y, mask=mask)


def fused_bn_sub_mean(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Computes out = (x - spatial_mean(x)[n,c]) * scale[c].
    This is exactly equivalent to BN(x) - spatial_mean(BN(x)) given
    scale = gamma / sqrt(var + eps).
    """
    N, C, D, H, W = x.shape
    S = D * H * W
    x = x.contiguous()
    out = torch.empty_like(x)
    sp_sum = torch.empty((N, C), device=x.device, dtype=torch.float32)
    sp_sumsq = torch.empty((N, C), device=x.device, dtype=torch.float32)
    grid = (N * C,)
    _reduce_nc_kernel[grid](x, sp_sum, sp_sumsq, N, C, S)
    sp_mean = sp_sum / S
    _apply_kernel[grid](x, out, sp_mean, scale, N, C, S)
    return out, sp_sum, sp_sumsq


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self._cached_scale = None

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape
        S = D * H * W
        bn = self.batch_norm
        eps = bn.eps

        if self.training:
            out, sp_sum, sp_sumsq = fused_bn_sub_mean_train(x, bn.weight, eps)
            # Update running stats
            with torch.no_grad():
                M = N * S
                # ch mean and var (biased for norm, unbiased for running)
                ch_sum = sp_sum.sum(dim=0)
                ch_sumsq = sp_sumsq.sum(dim=0)
                ch_mean = ch_sum / M
                ch_var_biased = ch_sumsq / M - ch_mean * ch_mean
                # unbiased
                if M > 1:
                    ch_var_unbiased = ch_var_biased * (M / (M - 1))
                else:
                    ch_var_unbiased = ch_var_biased
                mom = bn.momentum if bn.momentum is not None else 0.0
                if bn.track_running_stats:
                    bn.running_mean.mul_(1 - mom).add_(ch_mean, alpha=mom)
                    bn.running_var.mul_(1 - mom).add_(ch_var_unbiased, alpha=mom)
                    bn.num_batches_tracked.add_(1)
            return out
        else:
            inv = torch.rsqrt(bn.running_var + eps)
            scale = (bn.weight * inv).contiguous()
            out, _, _ = fused_bn_sub_mean(x, scale)
            return out


def fused_bn_sub_mean_train(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """Training-mode fused BN + sub_mean.
    out[n,c,s] = (x[n,c,s] - spatial_mean_x[n,c]) * scale[c]
    where scale[c] = weight[c] / sqrt(batch_var[c] + eps), with biased batch var.
    """
    N, C, D, H, W = x.shape
    S = D * H * W
    M = N * S
    x = x.contiguous()
    sp_sum = torch.empty((N, C), device=x.device, dtype=torch.float32)
    sp_sumsq = torch.empty((N, C), device=x.device, dtype=torch.float32)
    grid = (N * C,)
    _reduce_nc_kernel[grid](x, sp_sum, sp_sumsq, N, C, S)
    sp_mean = sp_sum / S
    # channel stats for BN scale
    ch_sum = sp_sum.sum(dim=0)
    ch_sumsq = sp_sumsq.sum(dim=0)
    ch_mean = ch_sum / M
    ch_var_biased = ch_sumsq / M - ch_mean * ch_mean
    scale = (weight / torch.sqrt(ch_var_biased + eps)).contiguous()
    out = torch.empty_like(x)
    _apply_kernel[grid](x, out, sp_mean, scale, N, C, S)
    return out, sp_sum, sp_sumsq