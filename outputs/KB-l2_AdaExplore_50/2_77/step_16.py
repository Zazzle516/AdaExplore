import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused: compute per-(N,C) sum over spatial dims, then apply affine.
# Output: (N, C) -> reshape to (N, C, 1, 1, 1)
@triton.jit
def _fused_mean_affine_kernel(
    y_ptr,         # (N*C, S)
    a_ptr,         # (C,)
    b_ptr,         # (C,)
    out_ptr,       # (N, C)
    N, C, S,
    inv_S,         # 1.0 / S as float
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = pid * S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    for s_off in range(0, S, BLOCK_S):
        offs = s_off + tl.arange(0, BLOCK_S)
        mask = offs < S
        v = tl.load(y_ptr + base + offs, mask=mask, other=0.0)
        acc += v
    total = tl.sum(acc, axis=0)
    mean = total * inv_S
    a = tl.load(a_ptr + c)
    b = tl.load(b_ptr + c)
    o = mean * a + b
    tl.store(out_ptr + n * C + c, o)


def _launch_fused(y, a, b):
    N, C, D, H, W = y.shape
    S = D * H * W
    y_flat = y.contiguous().view(N * C, S)
    out = torch.empty((N, C), device=y.device, dtype=y.dtype)

    # Choose BLOCK_S
    if S <= 2048:
        BLOCK_S = triton.next_power_of_2(S)
        num_warps = 4
    elif S <= 8192:
        BLOCK_S = 2048
        num_warps = 8
    else:
        BLOCK_S = 4096
        num_warps = 8

    grid = (N * C,)
    _fused_mean_affine_kernel[grid](
        y_flat, a, b, out,
        N, C, S, 1.0 / S,
        BLOCK_S=BLOCK_S, num_warps=num_warps, num_stages=2,
    )
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
            a = (s * gamma * inv).contiguous()
            b = (beta - mu * gamma * inv).contiguous()
            out = _launch_fused(y, a, b)
            return out.view(N, C, 1, 1, 1)
        else:
            running_mean = self.batch_norm.running_mean
            running_var = self.batch_norm.running_var
            inv = torch.rsqrt(running_var + self.eps)
            a = (s * gamma * inv).contiguous()
            b = (beta - running_mean * gamma * inv).contiguous()
            out = _launch_fused(y, a, b)
            return out.view(N, C, 1, 1, 1)