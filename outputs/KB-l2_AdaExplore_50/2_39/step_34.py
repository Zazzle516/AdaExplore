import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Persistent per-column-tile stats kernel: one program per N tile,
# loops over M and accumulates sum and sumsq of (y * scale).
@triton.jit
def scale_bn_stats_persistent_kernel(
    Y_ptr, scale_ptr, sum_ptr, sumsq_ptr,
    M, N,
    BLOCK_N: tl.constexpr, TILE_M: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    s = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)

    acc_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc_sumsq = tl.zeros([BLOCK_N], dtype=tl.float32)

    for m_start in range(0, M, TILE_M):
        offs_m = m_start + tl.arange(0, TILE_M)
        mask_m = offs_m < M
        ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
        y = tl.load(ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        v = y * s[None, :]
        acc_sum += tl.sum(v, axis=0)
        acc_sumsq += tl.sum(v * v, axis=0)

    tl.store(sum_ptr + offs_n, acc_sum, mask=mask_n)
    tl.store(sumsq_ptr + offs_n, acc_sumsq, mask=mask_n)


# Stage 2: apply out = y * a + b
@triton.jit
def fused_scale_bn_apply_kernel(
    Y_ptr, Out_ptr, a_ptr, b_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    a = tl.load(a_ptr + offs_n, mask=mask_n, other=0.0)
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)

    m_start = pid_m * BLOCK_M
    for i in tl.static_range(0, BLOCK_M, 32):
        offs_m = m_start + i + tl.arange(0, 32)
        mask_m = offs_m < M
        mask = mask_m[:, None] & mask_n[None, :]
        ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
        y = tl.load(ptrs, mask=mask, other=0.0)
        out = y * a[None, :] + b[None, :]
        out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(out_ptrs, out, mask=mask)


def fused_scale_batchnorm(y, scale, gamma, beta, running_mean, running_var, eps, momentum, training):
    M, N = y.shape
    if training:
        BLOCK_N_STAT = 128
        TILE_M = 32
        sum_buf = torch.empty(N, device=y.device, dtype=torch.float32)
        sumsq_buf = torch.empty(N, device=y.device, dtype=torch.float32)
        grid_stat = (triton.cdiv(N, BLOCK_N_STAT),)
        scale_bn_stats_persistent_kernel[grid_stat](
            y, scale, sum_buf, sumsq_buf,
            M, N,
            BLOCK_N=BLOCK_N_STAT, TILE_M=TILE_M,
            num_warps=8, num_stages=4,
        )

        mean = sum_buf / M
        var = sumsq_buf / M - mean * mean
        invstd = 1.0 / torch.sqrt(var + eps)

        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
            unbiased_var = var * (M / (M - 1)) if M > 1 else var
            running_var.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)
    else:
        mean = running_mean
        invstd = 1.0 / torch.sqrt(running_var + eps)

    a = scale * invstd * gamma
    b = beta - mean * invstd * gamma

    out = torch.empty_like(y)
    BLOCK_M_AP = 256
    BLOCK_N_AP = 128
    grid = (triton.cdiv(M, BLOCK_M_AP), triton.cdiv(N, BLOCK_N_AP))
    fused_scale_bn_apply_kernel[grid](
        y, out, a, b,
        M, N,
        BLOCK_M=BLOCK_M_AP, BLOCK_N=BLOCK_N_AP,
        num_warps=8, num_stages=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.eps = eps
        self.momentum = momentum
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.gemm.weight
        b = self.gemm.bias
        scale = self.scale.contiguous().view(-1)

        y = torch.addmm(b, x, W.t())

        out = fused_scale_batchnorm(
            y, scale,
            self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var,
            self.eps, self.momentum, self.training,
        )
        return out