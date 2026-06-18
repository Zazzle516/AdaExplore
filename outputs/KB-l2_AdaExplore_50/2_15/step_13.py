import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bn_mean_sub_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,  # [C]
    shift_ptr,  # [C]
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*C
    n = pid // C
    c = pid % C
    row_offset = pid * S

    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    # Pass 1: compute sum of (scale*x + shift)
    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        y = vals * scale + shift
        acc += tl.sum(tl.where(mask, y, 0.0), axis=0)
    mean = acc / S

    # Pass 2: write (y - mean)
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        y = vals * scale + shift - mean
        tl.store(out_ptr + row_offset + offs, y, mask=mask)


def bn_mean_sub(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    if S <= 2048:
        BLOCK_S = 1024
        num_warps = 4
    elif S <= 8192:
        BLOCK_S = 2048
        num_warps = 8
    else:
        BLOCK_S = 4096
        num_warps = 8
    _bn_mean_sub_kernel[grid](
        x, out, scale, shift, N, C, S, BLOCK_S=BLOCK_S, num_warps=num_warps
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
        N, C, D, H, W = x.shape
        bn = self.batch_norm
        eps = bn.eps

        if bn.training:
            # Compute batch stats per channel
            xf = x.reshape(N, C, -1)
            # mean & var across N and spatial
            # Use unbiased=False (BN convention for normalization)
            ch_mean = xf.mean(dim=(0, 2))
            ch_var = xf.var(dim=(0, 2), unbiased=False)
            # Update running stats
            with torch.no_grad():
                m = bn.momentum if bn.momentum is not None else 0.1
                if bn.track_running_stats and bn.running_mean is not None:
                    bn.running_mean.mul_(1 - m).add_(ch_mean.detach(), alpha=m)
                    # BN uses unbiased var for running_var update
                    n_elem = xf.shape[0] * xf.shape[2]
                    unbiased_var = ch_var.detach() * (n_elem / max(n_elem - 1, 1))
                    bn.running_var.mul_(1 - m).add_(unbiased_var, alpha=m)
                    bn.num_batches_tracked.add_(1)
            mean = ch_mean
            var = ch_var
        else:
            mean = bn.running_mean
            var = bn.running_var

        w = bn.weight if bn.weight is not None else torch.ones(C, device=x.device, dtype=x.dtype)
        b = bn.bias if bn.bias is not None else torch.zeros(C, device=x.device, dtype=x.dtype)
        invstd = torch.rsqrt(var + eps)
        scale = (w * invstd).contiguous()
        shift = (b - mean * w * invstd).contiguous()
        return bn_mean_sub(x, scale, shift)