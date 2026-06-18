import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


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
def _bn_sub_mean_eval_kernel(
    x_ptr, out_ptr,
    scale_ptr, bias_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = (n * C + c) * S

    scale = tl.load(scale_ptr + c)
    bias = tl.load(bias_ptr + c)

    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * scale + bias
        acc += tl.sum(tl.where(mask, y, 0.0), axis=0)
    mean = acc / S

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * scale + bias - mean
        tl.store(out_ptr + base + offs, y, mask=mask)


@triton.jit
def _per_channel_var_kernel(
    x_ptr,        # [N, C, S]
    var_ptr,      # [C]
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    # one program per channel; reduce over N*S
    c = tl.program_id(0)
    total = N * S
    sum_x = 0.0
    sum_x2 = 0.0
    # iterate over all (n, s) for this channel
    NS = N * S
    for off_start in range(0, NS, BLOCK_S):
        offs = off_start + tl.arange(0, BLOCK_S)
        mask = offs < NS
        n = offs // S
        s = offs % S
        idx = (n * C + c) * S + s
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        sum_x2 += tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    mean = sum_x / NS
    var = sum_x2 / NS - mean * mean
    tl.store(var_ptr + c, var)


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
def _train_fused_kernel(
    x_ptr,        # [N, C, S]
    out_ptr,      # [N, C, S]
    scale_ptr,    # [C] = gamma / sqrt(var+eps)
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = (n * C + c) * S

    scale = tl.load(scale_ptr + c)

    # compute per-(n,c) spatial mean of x
    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(tl.where(mask, x, 0.0), axis=0)
    x_mean = acc / S

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = (x - x_mean) * scale
        tl.store(out_ptr + base + offs, y, mask=mask)


def bn_sub_mean_eval(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, C, D, H, W = x.shape
    S = D * H * W
    x = x.contiguous()
    out = torch.empty_like(x)
    grid = (N * C,)
    _bn_sub_mean_eval_kernel[grid](x, out, scale, bias, N, C, S)
    return out


def bn_sub_mean_train(x: torch.Tensor, gamma: torch.Tensor, eps: float):
    """Returns (output, batch_mean, batch_var) for BN training stats update."""
    N, C, D, H, W = x.shape
    S = D * H * W
    x = x.contiguous()

    # compute per-channel var via kernel (also need mean for running stats)
    # easier: use torch for stats (small reduction), use kernel for fused output
    x_flat = x.view(N, C, S)
    # per-channel batch mean and var over (N, S)
    # using torch: shape [C]
    batch_mean = x_flat.mean(dim=(0, 2))
    batch_var = x_flat.var(dim=(0, 2), unbiased=False)

    scale = gamma / torch.sqrt(batch_var + eps)

    out = torch.empty_like(x)
    grid = (N * C,)
    _train_fused_kernel[grid](x, out, scale.contiguous(), N, C, S)
    return out, batch_mean, batch_var


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
        bn = self.batch_norm

        if self.training:
            out, batch_mean, batch_var = bn_sub_mean_train(x, bn.weight, bn.eps)
            # update running stats to match reference behavior
            if bn.track_running_stats and bn.running_mean is not None:
                with torch.no_grad():
                    momentum = bn.momentum if bn.momentum is not None else 0.1
                    N, C, D, H, W = x.shape
                    n_elem = N * D * H * W
                    unbiased_var = batch_var * (n_elem / max(n_elem - 1, 1))
                    bn.running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                    bn.running_var.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)
                    bn.num_batches_tracked.add_(1)
            return out
        else:
            eps = bn.eps
            inv = torch.rsqrt(bn.running_var + eps)
            scale = bn.weight * inv
            bias = bn.bias - bn.running_mean * scale
            return bn_sub_mean_eval(x, scale.contiguous(), bias.contiguous())