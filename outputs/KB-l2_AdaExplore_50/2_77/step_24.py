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


@triton.jit
def _gap_only_kernel(
    x_ptr, out_ptr,
    N, C, S,
    inv_S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * S
    acc = 0.0
    for s_off in range(0, S, BLOCK_S):
        offs = s_off + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    tl.store(out_ptr + pid, acc * inv_S)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.out_channels = out_channels
        self.eps = eps
        self._cached_w = None
        self._cached_b = None
        self._cache_version = -1

    def _get_folded_conv_params(self):
        # Fold scale_factor and BN affine into conv weight/bias at eval time.
        # conv weight shape: (in_channels, out_channels, kD, kH, kW)
        # After conv: y_c = sum_{ic,...} W[ic,c,...] * x[ic,...] + b[c]
        # Apply scale * BN: z_c = (s*y_c - mean_c) * gamma_c * rstd_c + beta_c
        #                       = y_c * (s*gamma*rstd) + (beta - mean*gamma*rstd)
        # Folding into conv: W'[ic,c,...] = W[ic,c,...] * (s*gamma_c*rstd_c)
        #                    b'[c] = b[c] * (s*gamma_c*rstd_c) + (beta_c - mean_c*gamma_c*rstd_c)
        bn = self.batch_norm
        ver = bn.running_mean._version + bn.running_var._version
        if (self._cached_w is None) or (ver != self._cache_version):
            s = self.scale_factor
            rstd = torch.rsqrt(bn.running_var + self.eps)
            gamma = bn.weight
            beta = bn.bias
            mean = bn.running_mean
            scale = s * gamma * rstd  # (out_channels,)
            shift = beta - mean * gamma * rstd  # (out_channels,)

            W = self.conv_transpose.weight  # (in, out, kD, kH, kW)
            b = self.conv_transpose.bias    # (out,)
            W_folded = W * scale.view(1, -1, 1, 1, 1)
            b_folded = b * scale + shift
            self._cached_w = W_folded.contiguous()
            self._cached_b = b_folded.contiguous()
            self._cache_version = ver
        return self._cached_w, self._cached_b

    def forward(self, x):
        if self.training:
            x = self.conv_transpose(x)
            x = x * self.scale_factor
            x = self.batch_norm(x)
            x = self.global_avg_pool(x)
            return x

        W, b = self._get_folded_conv_params()
        x = F.conv_transpose3d(x, W, b,
                               stride=self.conv_transpose.stride,
                               padding=self.conv_transpose.padding,
                               output_padding=self.conv_transpose.output_padding,
                               groups=self.conv_transpose.groups,
                               dilation=self.conv_transpose.dilation)

        x = x.contiguous()
        N, C, D, H, W_ = x.shape
        S = D * H * W_

        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)

        BLOCK_S = 2048
        grid = (N * C,)
        _gap_only_kernel[grid](
            x, out,
            N, C, S,
            1.0 / S,
            BLOCK_S=BLOCK_S,
            num_warps=8,
            num_stages=2,
        )
        return out