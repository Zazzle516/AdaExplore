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
def compute_ab_kernel(
    sum_ptr, sumsq_ptr, scale_ptr, gamma_ptr, beta_ptr,
    running_mean_ptr, running_var_ptr,
    a_ptr, b_ptr,
    M, N, eps, momentum,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    s = tl.load(sum_ptr + offs, mask=mask, other=0.0)
    sq = tl.load(sumsq_ptr + offs, mask=mask, other=0.0)
    scl = tl.load(scale_ptr + offs, mask=mask, other=0.0)
    g = tl.load(gamma_ptr + offs, mask=mask, other=0.0)
    bt = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    rm = tl.load(running_mean_ptr + offs, mask=mask, other=0.0)
    rv = tl.load(running_var_ptr + offs, mask=mask, other=0.0)

    Mf = M.to(tl.float32)
    mean = s / Mf
    var = sq / Mf - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    a = scl * invstd * g
    b = bt - mean * invstd * g

    # Update running stats
    new_rm = rm * (1.0 - momentum) + mean * momentum
    unbiased_var = var * (Mf / (Mf - 1.0))
    new_rv = rv * (1.0 - momentum) + unbiased_var * momentum

    tl.store(a_ptr + offs, a, mask=mask)
    tl.store(b_ptr + offs, b, mask=mask)
    tl.store(running_mean_ptr + offs, new_rm, mask=mask)
    tl.store(running_var_ptr + offs, new_rv, mask=mask)


@triton.jit
def compute_ab_eval_kernel(
    scale_ptr, gamma_ptr, beta_ptr,
    running_mean_ptr, running_var_ptr,
    a_ptr, b_ptr,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    scl = tl.load(scale_ptr + offs, mask=mask, other=0.0)
    g = tl.load(gamma_ptr + offs, mask=mask, other=0.0)
    bt = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    rm = tl.load(running_mean_ptr + offs, mask=mask, other=0.0)
    rv = tl.load(running_var_ptr + offs, mask=mask, other=0.0)
    invstd = 1.0 / tl.sqrt(rv + eps)
    a = scl * invstd * g
    b = bt - rm * invstd * g
    tl.store(a_ptr + offs, a, mask=mask)
    tl.store(b_ptr + offs, b, mask=mask)


@triton.jit
def fused_scale_bn_apply_inplace_kernel(
    Y_ptr, a_ptr, b_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
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
    a_buf = torch.empty(N, device=y.device, dtype=torch.float32)
    b_buf = torch.empty(N, device=y.device, dtype=torch.float32)

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
        BLOCK_N_AB = 128
        grid_ab = (triton.cdiv(N, BLOCK_N_AB),)
        compute_ab_kernel[grid_ab](
            sum_buf, sumsq_buf, scale, gamma, beta,
            running_mean, running_var,
            a_buf, b_buf,
            M, N, eps, momentum,
            BLOCK_N=BLOCK_N_AB,
            num_warps=4,
        )
    else:
        BLOCK_N_AB = 128
        grid_ab = (triton.cdiv(N, BLOCK_N_AB),)
        compute_ab_eval_kernel[grid_ab](
            scale, gamma, beta,
            running_mean, running_var,
            a_buf, b_buf,
            N, eps,
            BLOCK_N=BLOCK_N_AB,
            num_warps=4,
        )

    BLOCK_M_AP = 128
    BLOCK_N_AP = 128
    grid = (triton.cdiv(M, BLOCK_M_AP), triton.cdiv(N, BLOCK_N_AP))
    fused_scale_bn_apply_inplace_kernel[grid](
        y, a_buf, b_buf,
        M, N,
        BLOCK_M=BLOCK_M_AP, BLOCK_N=BLOCK_N_AP,
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

        y = torch.addmm(b, x, W.t())

        out = fused_scale_batchnorm(
            y, scale,
            self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var,
            self.eps, self.momentum, self.training,
        )
        return out