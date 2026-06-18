import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 8192}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 8192}, num_warps=16, num_stages=2),
    ],
    key=['C', 'S'],
)
@triton.jit
def _scale_bn_gap_kernel(
    x_ptr, out_ptr, scale_ptr, bias_ptr,
    N, C, S,
    scale_factor,
    BLOCK_S: tl.constexpr,
):
    # program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    s = tl.load(scale_ptr + c)
    b = tl.load(bias_ptr + c)

    eff_scale = scale_factor * s
    base = (n * C + c) * S
    acc = 0.0
    for off in range(0, S, BLOCK_S):
        idx = off + tl.arange(0, BLOCK_S)
        mask = idx < S
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        v = v * eff_scale
        acc += tl.sum(v)

    out = acc / S + b
    tl.store(out_ptr + n * C + c, out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = float(scale_factor)
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.out_channels = out_channels
        self.eps = eps

    def forward(self, x):
        x = self.conv_transpose(x)

        if self.training:
            # Fallback to reference path during training to keep BN running stats correct
            x = x * self.scale_factor
            x = self.batch_norm(x)
            x = self.global_avg_pool(x)
            return x

        N, C, D, H, W = x.shape
        S = D * H * W

        rm = self.batch_norm.running_mean
        rv = self.batch_norm.running_var
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        eps = self.batch_norm.eps

        invstd = torch.rsqrt(rv + eps)
        # y = (x*sf - rm) * invstd * gamma + beta
        #   = x * (sf * invstd * gamma) + (beta - rm * invstd * gamma)
        scale = invstd * gamma
        bias = beta - rm * invstd * gamma

        x = x.contiguous()
        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)

        grid = (N * C,)
        _scale_bn_gap_kernel[grid](
            x, out, scale, bias,
            N, C, S,
            self.scale_factor,
        )
        return out