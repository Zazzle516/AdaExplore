import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=4),
    ],
    key=['C', 'S'],
)
@triton.jit
def _gap_affine_kernel(
    x_ptr,         # [N, C, S]
    out_ptr,       # [N, C]
    scale_ptr,     # [C]
    bias_ptr,      # [C]
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * S + c * S
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for s_off in range(0, S, BLOCK_S):
        offs = s_off + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.where(mask, x, 0.0)

    total = tl.sum(acc, axis=0)
    inv_s = 1.0 / S.to(tl.float32)
    mean = total * inv_s

    scale = tl.load(scale_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    out = mean * scale + bias
    tl.store(out_ptr + n * C + c, out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.eps = eps

    def forward(self, x):
        x = self.conv_transpose(x)

        if self.training:
            # Fused training path: compute per-(N,C) mean and per-C mean/var of x,
            # then apply BN+GAP using batch stats analytically. Also update running stats.
            sf = self.scale_factor
            eps = self.eps
            momentum = self.batch_norm.momentum
            gamma = self.batch_norm.weight
            beta = self.batch_norm.bias

            N, C, D, H, W = x.shape
            S = D * H * W
            x_flat = x.contiguous().view(N, C, S)

            # per-(N,C) sum and per-(N,C) sum of squares
            sum_nc = x_flat.sum(dim=2)              # [N, C]
            sumsq_nc = (x_flat * x_flat).sum(dim=2) # [N, C]
            mean_nc = sum_nc / S                    # [N, C]  mean of x per (n,c)

            sum_c = sum_nc.sum(dim=0)               # [C]
            sumsq_c = sumsq_nc.sum(dim=0)           # [C]
            count = N * S
            mean_c_x = sum_c / count                # E[x] per c
            var_c_x = sumsq_c / count - mean_c_x * mean_c_x  # biased var of x per c

            # BN uses scaled values: y_pre = x*sf
            mu = sf * mean_c_x
            var = (sf * sf) * var_c_x
            inv_std = torch.rsqrt(var + eps)

            # update running stats (unbiased var)
            with torch.no_grad():
                unbiased_var = var * (count / max(count - 1, 1))
                self.batch_norm.running_mean.mul_(1 - momentum).add_(mu.detach(), alpha=momentum)
                self.batch_norm.running_var.mul_(1 - momentum).add_(unbiased_var.detach(), alpha=momentum)

            # output[n,c] = (mean_nc * sf - mu) * inv_std * gamma + beta
            scale = sf * inv_std * gamma            # [C]
            bias = beta - mu * inv_std * gamma      # [C]
            out = mean_nc * scale.unsqueeze(0) + bias.unsqueeze(0)
            return out.view(N, C, 1, 1, 1)

        rm = self.batch_norm.running_mean
        rv = self.batch_norm.running_var
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        inv = torch.rsqrt(rv + self.eps)
        A = (self.scale_factor * gamma * inv).contiguous()
        B = (beta - rm * gamma * inv).contiguous()

        N, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.contiguous().view(N, C, S)
        out = torch.empty((N, C), device=x.device, dtype=x.dtype)

        grid = (N * C,)
        _gap_affine_kernel[grid](
            x_flat, out, A, B,
            N, C, S,
        )
        return out.view(N, C, 1, 1, 1)