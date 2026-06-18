import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def scale_bn_stats_kernel(
    Y_ptr, scale_ptr, sum_ptr, sumsq_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m_chunk = tl.program_id(1)
    num_m_chunks = tl.num_programs(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    s = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)

    chunk_size = tl.cdiv(M, num_m_chunks)
    m_start_global = pid_m_chunk * chunk_size
    m_end_global = tl.minimum(m_start_global + chunk_size, M)

    sum_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sumsq_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    offs_m = tl.arange(0, BLOCK_M)
    for m_start in range(m_start_global, m_end_global, BLOCK_M):
        idx = m_start + offs_m
        mask_m = idx < m_end_global
        ptrs = Y_ptr + idx[:, None] * N + offs_n[None, :]
        y = tl.load(ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        v = y * s[None, :]
        sum_acc += tl.sum(v, axis=0)
        sumsq_acc += tl.sum(v * v, axis=0)

    tl.atomic_add(sum_ptr + offs_n, sum_acc, mask=mask_n)
    tl.atomic_add(sumsq_ptr + offs_n, sumsq_acc, mask=mask_n)


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
def fused_scale_bn_apply_inplace_kernel(
    Y_ptr, a_ptr, b_ptr,
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
    tl.store(ptrs, out, mask=mask)


@triton.jit
def fused_addmm_epilogue_kernel(
    Y_ptr, a_ptr, b_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Simple elementwise: Y = Y * a[col] + b[col], 2D tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
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
    tl.store(ptrs, out, mask=mask)


def fused_scale_batchnorm(y, scale, gamma, beta, running_mean, running_var, eps, momentum, training):
    M, N = y.shape
    if training:
        sum_buf = torch.zeros(N, device=y.device, dtype=torch.float32)
        sumsq_buf = torch.zeros(N, device=y.device, dtype=torch.float32)
        BLOCK_M_STAT = 512
        BLOCK_N_STAT = 128
        M_CHUNKS = 8
        grid_stat = (triton.cdiv(N, BLOCK_N_STAT), M_CHUNKS)
        scale_bn_stats_kernel[grid_stat](
            y, scale, sum_buf, sumsq_buf,
            M, N,
            BLOCK_M=BLOCK_M_STAT, BLOCK_N=BLOCK_N_STAT,
            num_warps=8, num_stages=3,
        )
        mean = sum_buf / M
        var = sumsq_buf / M - mean * mean
        invstd = 1.0 / torch.sqrt(var + eps)

        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
            unbiased_var = var * (M / (M - 1)) if M > 1 else var
            running_var.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)

        a = scale * invstd * gamma
        b = beta - mean * invstd * gamma

        # In-place apply: y already exists, write back to y
        BLOCK_M_AP = 128
        BLOCK_N_AP = 128
        GROUP_M = 8
        grid = (triton.cdiv(M, BLOCK_M_AP) * triton.cdiv(N, BLOCK_N_AP),)
        fused_scale_bn_apply_inplace_kernel[grid](
            y, a, b,
            M, N,
            BLOCK_M=BLOCK_M_AP, BLOCK_N=BLOCK_N_AP, GROUP_M=GROUP_M,
            num_warps=8, num_stages=3,
        )
        return y
    else:
        mean = running_mean
        invstd = 1.0 / torch.sqrt(running_var + eps)
        a = scale * invstd * gamma
        b = beta - mean * invstd * gamma
        BLOCK_M_AP = 128
        BLOCK_N_AP = 128
        GROUP_M = 8
        grid = (triton.cdiv(M, BLOCK_M_AP) * triton.cdiv(N, BLOCK_N_AP),)
        fused_scale_bn_apply_inplace_kernel[grid](
            y, a, b,
            M, N,
            BLOCK_M=BLOCK_M_AP, BLOCK_N=BLOCK_N_AP, GROUP_M=GROUP_M,
            num_warps=8, num_stages=3,
        )
        return y


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

        # heavy GEMM via cuBLAS
        y = torch.addmm(b, x, W.t())

        out = fused_scale_batchnorm(
            y, scale,
            self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var,
            self.eps, self.momentum, self.training,
        )
        return out