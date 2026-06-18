import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def col_stats_kernel(
    y_ptr, mean_ptr, var_ptr,
    M, N,
    inv_M,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)  # column index
    if pid < N:
        offs_m = tl.arange(0, BLOCK_M)
        s = 0.0
        sq = 0.0
        for m_start in range(0, M, BLOCK_M):
            idx = m_start + offs_m
            mask = idx < M
            v = tl.load(y_ptr + idx * stride_ym + pid * stride_yn, mask=mask, other=0.0)
            v = v.to(tl.float32)
            s += tl.sum(v, axis=0)
            sq += tl.sum(v * v, axis=0)
        mean = s * inv_M
        var = sq * inv_M - mean * mean
        tl.store(mean_ptr + pid, mean)
        tl.store(var_ptr + pid, var)


@triton.jit
def fused_bn_swish_kernel(
    y_ptr, scale_ptr, shift_ptr, bias_ptr, out_ptr,
    M, N,
    inv_div,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = M * N
    mask = offs < total
    col = offs % N
    b = tl.load(bias_ptr)
    v = tl.load(y_ptr + offs, mask=mask, other=0.0)
    s = tl.load(scale_ptr + col, mask=mask, other=0.0)
    sh = tl.load(shift_ptr + col, mask=mask, other=0.0)
    y = v * s + sh + b
    y = y * inv_div
    y = y * tl.sigmoid(y)
    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.divide_value = float(divide_value)

        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        # Linear via cuBLAS (highly tuned for these sizes)
        y = torch.addmm(self.matmul.bias, x, self.matmul.weight.t())  # (M, N)

        if self.training:
            # Compute batch mean/var via fused Triton reduction
            mean = torch.empty(N, device=y.device, dtype=torch.float32)
            var = torch.empty(N, device=y.device, dtype=torch.float32)
            BLOCK_M = 1024
            # ensure BLOCK_M is power of 2 and >= reasonable
            grid_stats = (N,)
            col_stats_kernel[grid_stats](
                y, mean, var,
                M, N,
                1.0 / M,
                y.stride(0), y.stride(1),
                BLOCK_M=BLOCK_M,
                num_warps=4,
                num_stages=2,
            )
            # Update running stats (use unbiased var for running_var as PyTorch does)
            with torch.no_grad():
                unbiased_var = var * (M / max(M - 1, 1))
                self.bn.running_mean.mul_(1 - self.bn_momentum).add_(mean, alpha=self.bn_momentum)
                self.bn.running_var.mul_(1 - self.bn_momentum).add_(unbiased_var, alpha=self.bn_momentum)
            inv_std = torch.rsqrt(var + self.bn_eps)
        else:
            mean = self.bn.running_mean
            var = self.bn.running_var
            inv_std = torch.rsqrt(var + self.bn_eps)

        scale = (self.bn.weight * inv_std).contiguous()
        shift = (self.bn.bias - mean * scale).contiguous()

        out = torch.empty_like(y)
        inv_div = 1.0 / self.divide_value
        total = M * N
        BLOCK = 8192
        grid = lambda meta: (triton.cdiv(total, meta['BLOCK']),)
        fused_bn_swish_kernel[grid](
            y, scale, shift, self.bias, out,
            M, N,
            inv_div,
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=2,
        )
        return out