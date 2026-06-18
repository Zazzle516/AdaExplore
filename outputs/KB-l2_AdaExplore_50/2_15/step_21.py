import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_scale_submean_kernel(
    x_ptr, out_ptr, scale_ptr,
    C, S,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    c = pid % C
    row_start = pid * S
    scale = tl.load(scale_ptr + c)

    # compute mean of x
    sum_val = tl.zeros((), dtype=tl.float32)
    for off in range(0, S, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
    mean = sum_val / S

    for off in range(0, S, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row_start + offs, scale * (x - mean), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape

        if self.training:
            # Compute batch mean/var across (N, D, H, W) per channel
            x_view = x.permute(1, 0, 2, 3, 4).contiguous().view(C, -1)
            n_elem = x_view.shape[1]
            batch_mean = x_view.mean(dim=1)
            batch_var = x_view.var(dim=1, unbiased=False)
            # Update running stats (BN uses unbiased var for running_var)
            with torch.no_grad():
                m = self.batch_norm.momentum
                self.batch_norm.running_mean.mul_(1 - m).add_(batch_mean, alpha=m)
                unbiased_var = batch_var * (n_elem / (n_elem - 1))
                self.batch_norm.running_var.mul_(1 - m).add_(unbiased_var, alpha=m)
                self.batch_norm.num_batches_tracked.add_(1)
            var = batch_var
            mean_c = batch_mean
        else:
            var = self.batch_norm.running_var
            mean_c = self.batch_norm.running_mean

        eps = self.batch_norm.eps
        gamma = self.batch_norm.weight
        # scale[c] = gamma[c] / sqrt(var[c] + eps)
        # After BN: y = scale[c] * (x - mean_c[c]) + beta[c]
        # spatial_mean_y[n,c] = scale[c]*(spatial_mean_x[n,c] - mean_c[c]) + beta[c]
        # y - spatial_mean_y = scale[c] * (x - spatial_mean_x[n,c])
        scale = gamma / torch.sqrt(var + eps)

        S = D * H * W
        x_flat = x.view(N * C, S)
        out = torch.empty_like(x_flat)
        grid = (N * C,)
        fused_scale_submean_kernel[grid](
            x_flat, out, scale.contiguous(),
            C, S, BLOCK_SIZE=1024, num_warps=4,
        )
        return out.view(N, C, D, H, W)