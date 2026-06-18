import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _sum_reduce_kernel(
    x_ptr, out_ptr,
    S,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * S
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s_start in range(0, S, BLOCK):
        offs = s_start + tl.arange(0, BLOCK)
        mask = offs < S
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += vals
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid, total)


@triton.jit
def _affine_kernel(
    mean_ptr, a_ptr, c_ptr, out_ptr,
    NC, C, inv_S,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NC
    c_idx = offs % C
    m = tl.load(mean_ptr + offs, mask=mask, other=0.0)
    a = tl.load(a_ptr + c_idx, mask=mask, other=0.0)
    cc = tl.load(c_ptr + c_idx, mask=mask, other=0.0)
    out = (m * inv_S) * a + cc
    tl.store(out_ptr + offs, out, mask=mask)


def triton_fused_avgpool_affine(y: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    N, C, D, H, W = y.shape
    S = D * H * W
    y_c = y.contiguous()
    sums = torch.empty((N * C,), device=y.device, dtype=torch.float32)
    BLOCK = 2048
    _sum_reduce_kernel[(N * C,)](y_c, sums, S, BLOCK=BLOCK, num_warps=8)

    out = torch.empty((N, C), device=y.device, dtype=y.dtype)
    BLOCK2 = 256
    NC = N * C
    grid2 = ((NC + BLOCK2 - 1) // BLOCK2,)
    _affine_kernel[grid2](sums, a, c, out, NC, C, 1.0 / S, BLOCK=BLOCK2, num_warps=2)
    return out.view(N, C, 1, 1, 1)


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
            c = (beta - mu * gamma * inv).contiguous()
            return triton_fused_avgpool_affine(y, a, c)
        else:
            bn = self.batch_norm
            inv = torch.rsqrt(bn.running_var + bn.eps)
            a = (s * gamma * inv).contiguous()
            c = (beta - bn.running_mean * gamma * inv).contiguous()
            return triton_fused_avgpool_affine(y, a, c)