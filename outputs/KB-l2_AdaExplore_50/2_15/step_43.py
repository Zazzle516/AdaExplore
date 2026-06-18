import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mean_sub_kernel(
    x_ptr,
    out_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    row_offset = pid * S

    # Compute mean
    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    mean = acc / S

    # Subtract mean
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row_offset + offs, vals - mean, mask=mask)


@triton.jit
def _scale_mean_sub_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    c = pid % C
    row_offset = pid * S
    scale = tl.load(scale_ptr + c)

    # Compute mean of x
    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    mean = acc / S

    # Output: scale * (x - mean)
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row_offset + offs, scale * (vals - mean), mask=mask)


def mean_sub(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    BLOCK_S = 4096
    num_warps = 8
    _mean_sub_kernel[grid](x, out, N, C, S, BLOCK_S=BLOCK_S, num_warps=num_warps, num_stages=2)
    return out


def scale_mean_sub(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    BLOCK_S = 1024
    num_warps = 4
    _scale_mean_sub_kernel[grid](
        x, scale, out, N, C, S,
        BLOCK_S=BLOCK_S, num_warps=num_warps, num_stages=2
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
        if self.training:
            # Match reference: BN computes batch stats and updates running stats.
            # Mathematically: bn(x) - mean_per_NC(bn(x)) = scale_c * (x - mean_per_NC(x))
            # where scale_c = weight_c / sqrt(batch_var_c + eps).
            # Compute batch_var manually so running stats update mirrors reference.
            bn = self.batch_norm
            # Per-channel batch mean/var over (N, D, H, W)
            with torch.no_grad():
                dims = (0, 2, 3, 4)
                batch_mean = x.mean(dim=dims)
                batch_var = x.var(dim=dims, unbiased=False)
                if bn.track_running_stats:
                    m = bn.momentum if bn.momentum is not None else 0.1
                    n_elems = x.numel() / x.size(1)
                    bn.running_mean.mul_(1 - m).add_(batch_mean, alpha=m)
                    # unbiased var for running stats
                    unbiased_var = batch_var * (n_elems / (n_elems - 1)) if n_elems > 1 else batch_var
                    bn.running_var.mul_(1 - m).add_(unbiased_var, alpha=m)
                    if bn.num_batches_tracked is not None:
                        bn.num_batches_tracked.add_(1)
            scale = bn.weight / torch.sqrt(batch_var + bn.eps)
            return scale_mean_sub(x, scale.contiguous())
        else:
            bn = self.batch_norm
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            return scale_mean_sub(x, scale.contiguous())