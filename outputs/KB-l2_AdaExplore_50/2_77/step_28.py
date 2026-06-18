import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mean_nc_kernel(
    y_ptr,
    out_ptr,
    M,
    stride_n, stride_c,
    BLOCK: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // tl.num_programs(1)  # not used pattern; instead use 2D grid
    # use 2d grid:
    pass


@triton.jit
def _mean_nc_2d_kernel(
    y_ptr,
    out_ptr,           # (N, C)
    M,                 # D*H*W
    C,
    BLOCK: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    base = (n * C + c) * M
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, M, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < M
        v = tl.load(y_ptr + base + idx, mask=mask, other=0.0)
        acc += v
    s = tl.sum(acc, axis=0)
    mean = s / M
    tl.store(out_ptr + n * C + c, mean)


@triton.jit
def _epilogue_kernel(
    nc_mean_ptr,   # (N, C)
    a_ptr,         # (C,)
    b_ptr,         # (C,)
    out_ptr,       # (N, C)
    N, C,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c_start = tl.program_id(1) * BLOCK_C
    offs = c_start + tl.arange(0, BLOCK_C)
    mask = offs < C
    m = tl.load(nc_mean_ptr + n * C + offs, mask=mask, other=0.0)
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    out = m * a + b
    tl.store(out_ptr + n * C + offs, out, mask=mask)


def fast_mean_nc(y: torch.Tensor) -> torch.Tensor:
    N, C, D, H, W = y.shape
    M = D * H * W
    out = torch.empty((N, C), device=y.device, dtype=torch.float32)
    BLOCK = 1024
    grid = (N, C)
    _mean_nc_2d_kernel[grid](y, out, M, C, BLOCK=BLOCK, num_warps=4, num_stages=2)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        y = self.conv_transpose(x)
        N, C, D, H, W = y.shape
        s = self.scale_factor

        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias

        if self.training:
            y_perm = y.permute(1, 0, 2, 3, 4).contiguous().view(C, -1)
            M = y_perm.shape[1]
            mu_y = y_perm.mean(dim=1)
            var_y = y_perm.var(dim=1, unbiased=False)

            mu = s * mu_y
            var = (s * s) * var_y

            with torch.no_grad():
                unbiased_var = var * (M / max(M - 1, 1))
                self.batch_norm.running_mean.mul_(1 - self.momentum).add_(mu.detach(), alpha=self.momentum)
                self.batch_norm.running_var.mul_(1 - self.momentum).add_(unbiased_var.detach(), alpha=self.momentum)

            inv = torch.rsqrt(var + self.eps)
            nc_mean = fast_mean_nc(y)
            a = s * gamma * inv
            b = beta - mu * gamma * inv
        else:
            running_mean = self.batch_norm.running_mean
            running_var = self.batch_norm.running_var
            inv = torch.rsqrt(running_var + self.eps)
            nc_mean = fast_mean_nc(y)
            a = s * gamma * inv
            b = beta - running_mean * gamma * inv

        out = torch.empty((N, C), device=y.device, dtype=torch.float32)
        BLOCK_C = 128
        grid = (N, triton.cdiv(C, BLOCK_C))
        _epilogue_kernel[grid](nc_mean, a, b, out, N, C, BLOCK_C=BLOCK_C, num_warps=4)
        return out.view(N, C, 1, 1, 1)