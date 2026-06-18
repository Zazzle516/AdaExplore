import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=2, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_scale_kernel(
    A, B, Bias, Scale, Y,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
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

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_am < M
    mask_n = offs_bn < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(Bias + offs_bn, mask=mask_n, other=0.0)
    scale = tl.load(Scale + offs_bn, mask=mask_n, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    y_ptrs = Y + (offs_am[:, None] * stride_ym + offs_bn[None, :] * stride_yn)
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def col_stats_kernel(
    Y,                 # [M, N]
    Mean, Var,         # [N]
    M, N,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    sum_x = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        vals = tl.load(ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        sum_x += tl.sum(vals, axis=0)
        sum_x2 += tl.sum(vals * vals, axis=0)

    inv_m = 1.0 / M
    mean = sum_x * inv_m
    var = sum_x2 * inv_m - mean * mean
    tl.store(Mean + offs_n, mean, mask=mask_n)
    tl.store(Var + offs_n, var, mask=mask_n)


@triton.jit
def bn_apply_kernel(
    Y, Out,
    Mean, Var,
    Gamma, Beta,
    M, N,
    stride_ym, stride_yn,
    stride_om, stride_on,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    mean = tl.load(Mean + offs_n, mask=mask_n, other=0.0)
    var = tl.load(Var + offs_n, mask=mask_n, other=0.0)
    gamma = tl.load(Gamma + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(Beta + offs_n, mask=mask_n, other=0.0)
    invstd = 1.0 / tl.sqrt(var + eps)
    scale = gamma * invstd
    shift = beta - mean * scale

    ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    vals = tl.load(ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    out = vals * scale[None, :] + shift[None, :]
    optrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(optrs, out, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def bn_apply_eval_kernel(
    Y, Out,
    RunMean, RunVar,
    Gamma, Beta,
    M, N,
    stride_ym, stride_yn,
    stride_om, stride_on,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    mean = tl.load(RunMean + offs_n, mask=mask_n, other=0.0)
    var = tl.load(RunVar + offs_n, mask=mask_n, other=0.0)
    gamma = tl.load(Gamma + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(Beta + offs_n, mask=mask_n, other=0.0)
    invstd = 1.0 / tl.sqrt(var + eps)
    scale = gamma * invstd
    shift = beta - mean * scale

    ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    vals = tl.load(ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    out = vals * scale[None, :] + shift[None, :]
    optrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(optrs, out, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = float(eps)
        self.momentum = float(momentum)

        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        W = self.gemm.weight  # [N, K]
        b = self.gemm.bias    # [N]
        scale = self.scale.view(-1).contiguous()

        Y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # Use W^T as B: A is x[M,K], B is W^T[K,N]
        # We'll do this by passing W with swapped strides.
        # W is [N, K] contiguous -> stride (K, 1). We want B of shape [K, N] with
        # stride_bk = 1, stride_bn = K (transpose view).
        stride_am, stride_ak = x.stride(0), x.stride(1)
        stride_bk, stride_bn = 1, K
        stride_ym, stride_yn = Y.stride(0), Y.stride(1)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        gemm_scale_kernel[grid](
            x, W, b, scale, Y,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_ym, stride_yn,
        )

        Out = torch.empty_like(Y)
        gamma = self.bn.weight
        beta = self.bn.bias

        if self.training:
            BLOCK_N_STAT = 128
            BLOCK_M_STAT = 128
            mean = torch.empty(N, device=x.device, dtype=torch.float32)
            var = torch.empty(N, device=x.device, dtype=torch.float32)
            grid_stats = (triton.cdiv(N, BLOCK_N_STAT),)
            col_stats_kernel[grid_stats](
                Y, mean, var,
                M, N,
                stride_ym, stride_yn,
                BLOCK_M=BLOCK_M_STAT, BLOCK_N=BLOCK_N_STAT,
            )

            # Update running stats (unbiased var for running)
            with torch.no_grad():
                self.bn.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                unbiased_var = var * (M / max(M - 1, 1))
                self.bn.running_var.mul_(1 - self.momentum).add_(unbiased_var, alpha=self.momentum)
                self.bn.num_batches_tracked.add_(1)

            BLOCK_M_APP = 64
            BLOCK_N_APP = 128
            grid_app = (triton.cdiv(M, BLOCK_M_APP), triton.cdiv(N, BLOCK_N_APP))
            bn_apply_kernel[grid_app](
                Y, Out,
                mean, var,
                gamma, beta,
                M, N,
                stride_ym, stride_yn,
                Out.stride(0), Out.stride(1),
                eps=self.eps,
                BLOCK_M=BLOCK_M_APP, BLOCK_N=BLOCK_N_APP,
            )
        else:
            BLOCK_M_APP = 64
            BLOCK_N_APP = 128
            grid_app = (triton.cdiv(M, BLOCK_M_APP), triton.cdiv(N, BLOCK_N_APP))
            bn_apply_eval_kernel[grid_app](
                Y, Out,
                self.bn.running_mean, self.bn.running_var,
                gamma, beta,
                M, N,
                stride_ym, stride_yn,
                Out.stride(0), Out.stride(1),
                eps=self.eps,
                BLOCK_M=BLOCK_M_APP, BLOCK_N=BLOCK_N_APP,
            )

        return Out