import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def scale_bn_stats_partial_kernel(
    Y_ptr, scale_ptr, sum_ptr, sumsq_ptr,
    M, N, NUM_M_CHUNKS,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    s = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)

    chunk_size = tl.cdiv(M, NUM_M_CHUNKS)
    m_begin = pid_m * chunk_size
    m_end = tl.minimum(m_begin + chunk_size, M)

    sum_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sumsq_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    offs_m = tl.arange(0, BLOCK_M)
    for m_start in range(m_begin, m_end, BLOCK_M):
        idx = m_start + offs_m
        mask_m = idx < m_end
        ptrs = Y_ptr + idx[:, None] * N + offs_n[None, :]
        y = tl.load(ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        v = y * s[None, :]
        sum_acc += tl.sum(v, axis=0)
        sumsq_acc += tl.sum(v * v, axis=0)
    out_off = pid_m * N + offs_n
    tl.store(sum_ptr + out_off, sum_acc, mask=mask_n)
    tl.store(sumsq_ptr + out_off, sumsq_acc, mask=mask_n)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'GROUP_M': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_scale_bn_apply_kernel(
    Y_ptr, Out_ptr, a_ptr, b_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    a = tl.load(a_ptr + offs_n, mask=mask_n, other=0.0)
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)

    ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    y = tl.load(ptrs, mask=mask, other=0.0)
    out = y * a[None, :] + b[None, :]
    out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def finalize_ab_kernel(
    sum_ptr, sumsq_ptr, scale_ptr, gamma_ptr, beta_ptr,
    a_ptr, b_ptr,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    s = tl.load(sum_ptr + offs, mask=mask, other=0.0)
    sq = tl.load(sumsq_ptr + offs, mask=mask, other=0.0)
    sc = tl.load(scale_ptr + offs, mask=mask, other=0.0)
    g = tl.load(gamma_ptr + offs, mask=mask, other=0.0)
    bt = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    inv_M = 1.0 / M
    mean = s * inv_M
    var = sq * inv_M - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)
    a = sc * invstd * g
    b = bt - mean * invstd * g
    tl.store(a_ptr + offs, a, mask=mask)
    tl.store(b_ptr + offs, b, mask=mask)


def fused_scale_batchnorm(y, scale, gamma, beta, running_mean, running_var, eps, momentum, training):
    M, N = y.shape
    if training:
        BLOCK_M_STAT = 512
        BLOCK_N_STAT = 128
        NUM_M_CHUNKS = 16
        sum_buf = torch.empty((NUM_M_CHUNKS, N), device=y.device, dtype=torch.float32)
        sumsq_buf = torch.empty((NUM_M_CHUNKS, N), device=y.device, dtype=torch.float32)
        grid_stat = (triton.cdiv(N, BLOCK_N_STAT), NUM_M_CHUNKS)
        scale_bn_stats_partial_kernel[grid_stat](
            y, scale, sum_buf, sumsq_buf,
            M, N, NUM_M_CHUNKS,
            BLOCK_M=BLOCK_M_STAT, BLOCK_N=BLOCK_N_STAT,
            num_warps=8, num_stages=3,
        )
        sum_total = sum_buf.sum(dim=0)
        sumsq_total = sumsq_buf.sum(dim=0)

        a = torch.empty(N, device=y.device, dtype=y.dtype)
        b = torch.empty(N, device=y.device, dtype=y.dtype)
        BLOCK_N_FIN = 256
        grid_fin = (triton.cdiv(N, BLOCK_N_FIN),)
        finalize_ab_kernel[grid_fin](
            sum_total, sumsq_total, scale, gamma, beta,
            a, b,
            M, N, eps,
            BLOCK_N=BLOCK_N_FIN,
            num_warps=4, num_stages=2,
        )

        with torch.no_grad():
            mean = sum_total / M
            var = sumsq_total / M - mean * mean
            running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
            unbiased_var = var * (M / (M - 1)) if M > 1 else var
            running_var.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)
    else:
        invstd = 1.0 / torch.sqrt(running_var + eps)
        a = scale * invstd * gamma
        b = beta - running_mean * invstd * gamma

    out = torch.empty_like(y)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    fused_scale_bn_apply_kernel[grid](
        y, out, a, b,
        M, N,
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
        W = self.gemm.weight
        b = self.gemm.bias

        y = torch.addmm(b, x, W.t())

        out = fused_scale_batchnorm(
            y, self.scale,
            self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var,
            self.eps, self.momentum, self.training,
        )
        return out