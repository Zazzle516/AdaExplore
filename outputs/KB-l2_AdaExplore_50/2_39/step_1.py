import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_scaled_kernel(
    A_ptr, B_ptr, bias_ptr, scale_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def gemm_scaled(x, weight, bias, scale):
    M, K = x.shape
    N, _ = weight.shape
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    gemm_scaled_kernel[grid](
        x, weight, bias, scale, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # weight is [N,K], we want B as [K,N], so transpose via strides
        out.stride(0), out.stride(1),
    )
    return out


# BatchNorm1d training: compute mean & var over batch dim per feature, then normalize
@triton.jit
def bn_stats_kernel(
    X_ptr, mean_ptr, var_ptr,
    M, N,
    stride_xm, stride_xn,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)  # feature index
    if pid >= N:
        return
    offs_m = tl.arange(0, BLOCK_M)
    sum_x = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        idx = m_start + offs_m
        mask = idx < M
        x = tl.load(X_ptr + idx * stride_xm + pid * stride_xn, mask=mask, other=0.0)
        sum_x += tl.where(mask, x, 0.0)
        sum_x2 += tl.where(mask, x * x, 0.0)
    s = tl.sum(sum_x, axis=0)
    s2 = tl.sum(sum_x2, axis=0)
    mean = s / M
    var = s2 / M - mean * mean
    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr + pid, var)


@triton.jit
def bn_apply_kernel(
    X_ptr, Y_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    M, N, eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = M * N
    mask = offs < total
    n_idx = offs % N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + n_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + n_idx, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + n_idx, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + n_idx, mask=mask, other=0.0)
    inv = 1.0 / tl.sqrt(var + eps)
    y = (x - mean) * inv * gamma + beta
    tl.store(Y_ptr + offs, y, mask=mask)


def batch_norm_train(x, gamma, beta, running_mean, running_var, eps, momentum):
    M, N = x.shape
    mean = torch.empty(N, device=x.device, dtype=torch.float32)
    var = torch.empty(N, device=x.device, dtype=torch.float32)

    BLOCK_M = 1024
    bn_stats_kernel[(N,)](
        x, mean, var, M, N,
        x.stride(0), x.stride(1),
        BLOCK_M=BLOCK_M,
    )

    out = torch.empty_like(x)
    total = M * N
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    bn_apply_kernel[grid](
        x, out, mean, var, gamma, beta,
        M, N, eps,
        BLOCK=BLOCK,
    )

    # update running stats (unbiased var for running_var)
    with torch.no_grad():
        running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
        unbiased_var = var * (M / (M - 1)) if M > 1 else var
        running_var.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)

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
        W = self.gemm.weight.contiguous()
        b = self.gemm.bias.contiguous()
        scale = self.scale.contiguous().view(-1)

        y = gemm_scaled(x, W, b, scale)

        if self.training:
            out = batch_norm_train(
                y,
                self.bn.weight, self.bn.bias,
                self.bn.running_mean, self.bn.running_var,
                self.eps, self.momentum,
            )
        else:
            # eval: fold into elementwise
            inv = 1.0 / torch.sqrt(self.bn.running_var + self.eps)
            scale2 = self.bn.weight * inv
            shift = self.bn.bias - self.bn.running_mean * scale2
            out = y * scale2 + shift
        return out