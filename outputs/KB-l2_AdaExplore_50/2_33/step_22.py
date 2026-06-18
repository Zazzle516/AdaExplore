import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_scale_stats_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr,
    Y_ptr, sum_ptr, sumsq_ptr,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < k_remaining) & mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=mask)

    # accumulate per-column sums into workspace via atomic add
    acc_masked = tl.where(mask, acc, 0.0)
    col_sum = tl.sum(acc_masked, axis=0)
    col_sumsq = tl.sum(acc_masked * acc_masked, axis=0)
    tl.atomic_add(sum_ptr + offs_n, col_sum, mask=mask_n)
    tl.atomic_add(sumsq_ptr + offs_n, col_sumsq, mask=mask_n)


@triton.jit
def finalize_stats_kernel(
    sum_ptr, sumsq_ptr,
    mean_ptr, invstd_ptr,
    running_mean_ptr, running_var_ptr,
    M, N, eps, momentum,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    s = tl.load(sum_ptr + offs, mask=mask, other=0.0)
    sq = tl.load(sumsq_ptr + offs, mask=mask, other=0.0)
    m = s / M
    var = sq / M - m * m
    var = tl.maximum(var, 0.0)
    invstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + offs, m, mask=mask)
    tl.store(invstd_ptr + offs, invstd, mask=mask)

    # update running stats: unbiased var for running
    rm = tl.load(running_mean_ptr + offs, mask=mask, other=0.0)
    rv = tl.load(running_var_ptr + offs, mask=mask, other=0.0)
    unbiased = var * (M / (M - 1))
    new_rm = (1 - momentum) * rm + momentum * m
    new_rv = (1 - momentum) * rv + momentum * unbiased
    tl.store(running_mean_ptr + offs, new_rm, mask=mask)
    tl.store(running_var_ptr + offs, new_rv, mask=mask)


@triton.jit
def bn_apply_kernel(
    Y_ptr, mean_ptr, invstd_ptr, gamma_ptr, beta_ptr,
    Out_ptr,
    M, N,
    stride_ym, stride_yn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    invstd = tl.load(invstd_ptr + offs_n, mask=mask_n, other=0.0)
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0)
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask = mask_m[:, None] & mask_n[None, :]
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    out = (y - mean[None, :]) * invstd[None, :] * gamma[None, :] + beta[None, :]
    o_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = x.contiguous()
        M, K = x.shape
        N = self.out_features
        device = x.device

        W = self.gemm.weight  # (N, K)
        Wt = W.t().contiguous()  # (K, N)
        bias = self.gemm.bias.contiguous()
        scale = self.scale.contiguous()

        Y = torch.empty((M, N), device=device, dtype=torch.float32)

        if self.training:
            sum_buf = torch.zeros(N, device=device, dtype=torch.float32)
            sumsq_buf = torch.zeros(N, device=device, dtype=torch.float32)

            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
            gemm_scale_stats_kernel[grid](
                x, Wt, bias, scale,
                Y, sum_buf, sumsq_buf,
                M, N, K,
                x.stride(0), x.stride(1),
                Wt.stride(0), Wt.stride(1),
                Y.stride(0), Y.stride(1),
            )

            mean = torch.empty(N, device=device, dtype=torch.float32)
            invstd = torch.empty(N, device=device, dtype=torch.float32)
            BLOCK_N_FIN = 256
            grid2 = (triton.cdiv(N, BLOCK_N_FIN),)
            finalize_stats_kernel[grid2](
                sum_buf, sumsq_buf,
                mean, invstd,
                self.bn.running_mean, self.bn.running_var,
                M, N, self.eps, self.momentum,
                BLOCK_N=BLOCK_N_FIN,
            )
            self.bn.num_batches_tracked += 1
        else:
            # just GEMM+scale, no stats needed
            sum_buf = torch.zeros(N, device=device, dtype=torch.float32)
            sumsq_buf = torch.zeros(N, device=device, dtype=torch.float32)
            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
            gemm_scale_stats_kernel[grid](
                x, Wt, bias, scale,
                Y, sum_buf, sumsq_buf,
                M, N, K,
                x.stride(0), x.stride(1),
                Wt.stride(0), Wt.stride(1),
                Y.stride(0), Y.stride(1),
            )
            mean = self.bn.running_mean
            invstd = 1.0 / torch.sqrt(self.bn.running_var + self.eps)

        Out = torch.empty((M, N), device=device, dtype=torch.float32)
        BM, BN = 64, 128
        grid3 = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        bn_apply_kernel[grid3](
            Y, mean, invstd, self.bn.weight, self.bn.bias,
            Out,
            M, N,
            Y.stride(0), Y.stride(1),
            Out.stride(0), Out.stride(1),
            BLOCK_M=BM, BLOCK_N=BN,
        )
        return Out