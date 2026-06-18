import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 4096},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_S': 8192},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_S': 8192},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_S': 16384}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_S': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_S': 32768}, num_warps=16, num_stages=2),
    ],
    key=['S'],
)
@triton.jit
def _gap_affine_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    bias_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * S + c * S
    acc = 0.0
    for s_off in range(0, S, BLOCK_S):
        offs = s_off + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)

    inv_s = 1.0 / S.to(tl.float32)
    mean = acc * inv_s

    scale = tl.load(scale_ptr + c)
    bias = tl.load(bias_ptr + c)
    out = mean * scale + bias
    tl.store(out_ptr + n * C + c, out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 4096},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_S': 8192},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_S': 8192},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_S': 16384}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_S': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_S': 32768}, num_warps=16, num_stages=2),
    ],
    key=['S'],
)
@triton.jit
def _gap_affine_kernel_cl(
    x_ptr,
    out_ptr,
    scale_ptr,
    bias_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    # x layout: (N, S, C) - channels_last_3d: contiguous in C
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * S * C + c
    acc = 0.0
    for s_off in range(0, S, BLOCK_S):
        offs = s_off + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs * C, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)

    inv_s = 1.0 / S.to(tl.float32)
    mean = acc * inv_s

    scale = tl.load(scale_ptr + c)
    bias = tl.load(bias_ptr + c)
    out = mean * scale + bias
    tl.store(out_ptr + n * C + c, out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        torch.backends.cudnn.benchmark = True
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.eps = eps
        self._cached_A = None
        self._cached_B = None
        self._cached_version = None

    def _get_AB(self):
        rm = self.batch_norm.running_mean
        rv = self.batch_norm.running_var
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        version = (rm._version, rv._version, gamma._version, beta._version)
        if self._cached_version != version or self._cached_A is None:
            inv = torch.rsqrt(rv + self.eps)
            A = (self.scale_factor * gamma * inv).contiguous()
            B = (beta - rm * gamma * inv).contiguous()
            self._cached_A = A
            self._cached_B = B
            self._cached_version = version
        return self._cached_A, self._cached_B

    def forward(self, x):
        if self.training:
            x = self.conv_transpose(x)
            x = x * self.scale_factor
            x = self.batch_norm(x)
            x = self.global_avg_pool(x)
            return x

        # Try channels_last_3d for faster cuDNN
        x_cl = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x_cl)

        A, B = self._get_AB()

        N, C, D, H, W = x.shape
        S = D * H * W
        out = torch.empty((N, C), device=x.device, dtype=x.dtype)

        if x.is_contiguous(memory_format=torch.channels_last_3d):
            # Layout: (N, D, H, W, C) memory order
            grid = (N * C,)
            _gap_affine_kernel_cl[grid](
                x, out, A, B,
                N, C, S,
            )
        else:
            x_flat = x.contiguous().view(N, C, S)
            grid = (N * C,)
            _gap_affine_kernel[grid](
                x_flat, out, A, B,
                N, C, S,
            )
        return out.view(N, C, 1, 1, 1)