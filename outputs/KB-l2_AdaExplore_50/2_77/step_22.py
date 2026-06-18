import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gap_kernel(
    x_ptr, out_ptr,
    weight_ptr, bias_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    w = tl.load(weight_ptr + c)
    b = tl.load(bias_ptr + c)

    base = (n * C + c) * S
    acc = 0.0
    for s_off in range(0, S, BLOCK_S):
        offs = s_off + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)

    mean = acc / S
    out_val = mean * w + b
    tl.store(out_ptr + n * C + c, out_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.out_channels = out_channels
        self.eps = eps

    def forward(self, x):
        x = self.conv_transpose(x)

        if self.training:
            # fall back to standard implementation for training
            x = x * self.scale_factor
            x = self.batch_norm(x)
            x = self.global_avg_pool(x)
            return x

        # Eval mode: fold scale, BN affine into single per-channel scale/bias,
        # then fuse with global average pool.
        # y = ((x * s - mean) / sqrt(var+eps)) * gamma + beta
        #   = x * (s * gamma / sqrt(var+eps)) + (beta - mean * gamma / sqrt(var+eps))
        s = self.scale_factor
        rstd = torch.rsqrt(self.batch_norm.running_var + self.eps)
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        mean = self.batch_norm.running_mean

        eff_w = (s * gamma * rstd).contiguous()
        eff_b = (beta - mean * gamma * rstd).contiguous()

        x = x.contiguous()
        N, C, D, H, W = x.shape
        S = D * H * W

        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)

        BLOCK_S = 1024
        grid = (N * C,)
        _gap_kernel[grid](
            x, out,
            eff_w, eff_b,
            N, C, S,
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )
        return out