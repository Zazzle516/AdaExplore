import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Kernel 1: per (n,c), compute sum and sum_sq over spatial dims.
@triton.jit
def _per_nc_stats_kernel(
    x_ptr,
    sum_ptr,     # [N, C]
    sumsq_ptr,   # [N, C]
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = (n * C + c) * S

    s_acc = 0.0
    sq_acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        s_acc += tl.sum(x, axis=0)
        sq_acc += tl.sum(x * x, axis=0)

    tl.store(sum_ptr + n * C + c, s_acc)
    tl.store(sumsq_ptr + n * C + c, sq_acc)


# Kernel 2: per (n,c), write out = scale[c] * (x - spatial_mean[n,c])
@triton.jit
def _apply_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,    # [C]
    mean_ptr,     # [N, C]  (spatial mean of x per (n,c))
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = (n * C + c) * S

    scale = tl.load(scale_ptr + c)
    mean = tl.load(mean_ptr + n * C + c)

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = scale * (x - mean)
        tl.store(out_ptr + base + offs, y, mask=mask)


def fused_bn_sub_mean(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
    momentum: float,
    training: bool,
):
    """
    Fuses BN + (subtract per-(n,c) spatial mean of BN output).

    Algebra (no heavy op affected; conv_transpose ran upstream as usual):
      BN(x)[n,c,*] = scale[c]*x[n,c,*] + shift[c]
      spatial_mean_per_nc(BN(x)) = scale[c]*spatial_mean_per_nc(x) + shift[c]
      BN(x) - spatial_mean_per_nc(BN(x)) = scale[c] * (x - spatial_mean_per_nc(x))
    where scale[c] = weight[c] / sqrt(var[c] + eps).

    In training mode we also update running_mean/running_var using the batch
    statistics of x (the BN input), matching nn.BatchNorm3d semantics.
    """
    assert x.is_cuda and x.dim() == 5
    N, C, D, H, W = x.shape
    S = D * H * W
    x = x.contiguous()
    out = torch.empty_like(x)

    BLOCK_S = 4096
    grid = (N * C,)

    if training:
        sum_nc = torch.empty((N, C), device=x.device, dtype=torch.float32)
        sumsq_nc = torch.empty((N, C), device=x.device, dtype=torch.float32)
        _per_nc_stats_kernel[grid](
            x, sum_nc, sumsq_nc, N, C, S,
            BLOCK_S=BLOCK_S, num_warps=4, num_stages=2,
        )
        spatial_mean = sum_nc / S  # [N, C]
        # Per-channel batch stats over N*S samples
        total = float(N * S)
        sum_c = sum_nc.sum(dim=0)
        sumsq_c = sumsq_nc.sum(dim=0)
        batch_mean = sum_c / total
        batch_var_biased = sumsq_c / total - batch_mean * batch_mean
        # Update running stats (use unbiased var for running_var, like PyTorch)
        with torch.no_grad():
            unbiased_var = batch_var_biased * (total / max(total - 1.0, 1.0))
            running_mean.mul_(1 - momentum).add_(batch_mean.detach(), alpha=momentum)
            running_var.mul_(1 - momentum).add_(unbiased_var.detach(), alpha=momentum)
        invstd = torch.rsqrt(batch_var_biased + eps)
        scale = (weight * invstd).contiguous()
    else:
        # Use running stats
        invstd = torch.rsqrt(running_var + eps)
        scale = (weight * invstd).contiguous()
        # Need spatial mean per (n,c) of x
        sum_nc = torch.empty((N, C), device=x.device, dtype=torch.float32)
        sumsq_nc = torch.empty((N, C), device=x.device, dtype=torch.float32)
        _per_nc_stats_kernel[grid](
            x, sum_nc, sumsq_nc, N, C, S,
            BLOCK_S=BLOCK_S, num_warps=4, num_stages=2,
        )
        spatial_mean = sum_nc / S

    spatial_mean = spatial_mean.contiguous()
    _apply_kernel[grid](
        x, out, scale, spatial_mean, N, C, S,
        BLOCK_S=BLOCK_S, num_warps=4, num_stages=2,
    )
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
        return fused_bn_sub_mean(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.eps,
            self.batch_norm.momentum,
            self.training,
        )