import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_scale_kernel(
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
        k_remain = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(mask_m[:, None]) & (offs_k[None, :] < k_remain), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remain) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)
    acc = (acc + bias[None, :]) * scale[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def gemm_scale(x, weight, bias, scale):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_scale_kernel[grid](
        x, weight, bias, scale, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # weight is [N,K], we want B[k,n]=weight[n,k]
        out.stride(0), out.stride(1),
    )
    return out


@triton.jit
def bn_stats_kernel(
    X_ptr, mean_ptr, invstd_ptr,
    M, N, eps,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per column
    n = pid
    offs = tl.arange(0, BLOCK_M)
    sum_x = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_M):
        idx = m_start + offs
        mask = idx < M
        x = tl.load(X_ptr + idx * N + n, mask=mask, other=0.0)
        sum_x += tl.where(mask, x, 0.0)
        sum_x2 += tl.where(mask, x * x, 0.0)
    s = tl.sum(sum_x, axis=0)
    s2 = tl.sum(sum_x2, axis=0)
    mean = s / M
    var = s2 / M - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + n, mean)
    tl.store(invstd_ptr + n, invstd)


@triton.jit
def bn_apply_kernel(
    X_ptr, Y_ptr, mean_ptr, invstd_ptr, weight_ptr, bias_ptr,
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

    ptrs = offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(X_ptr + ptrs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    invstd = tl.load(invstd_ptr + offs_n, mask=mask_n, other=1.0)
    w = tl.load(weight_ptr + offs_n, mask=mask_n, other=1.0)
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    y = (x - mean[None, :]) * invstd[None, :] * w[None, :] + b[None, :]
    tl.store(Y_ptr + ptrs, y, mask=mask)


def batchnorm_train(x, bn_weight, bn_bias, running_mean, running_var, eps, momentum):
    M, N = x.shape
    mean = torch.empty(N, device=x.device, dtype=torch.float32)
    invstd = torch.empty(N, device=x.device, dtype=torch.float32)
    BLOCK_M_STAT = 1024
    bn_stats_kernel[(N,)](x, mean, invstd, M, N, eps, BLOCK_M=BLOCK_M_STAT)

    out = torch.empty_like(x)
    BLOCK_M = 64
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    bn_apply_kernel[grid](x, out, mean, invstd, bn_weight, bn_bias, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    # update running stats
    with torch.no_grad():
        var_unbiased = (1.0 / (invstd * invstd) - eps) * (M / max(M - 1, 1))
        var_biased = (1.0 / (invstd * invstd) - eps)
        running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
        running_var.mul_(1 - momentum).add_(var_unbiased, alpha=momentum)
    return out


def batchnorm_eval(x, bn_weight, bn_bias, running_mean, running_var, eps):
    M, N = x.shape
    invstd = 1.0 / torch.sqrt(running_var + eps)
    out = torch.empty_like(x)
    BLOCK_M = 64
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    bn_apply_kernel[grid](x, out, running_mean, invstd, bn_weight, bn_bias, M, N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
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
        weight = self.gemm.weight.contiguous()
        bias = self.gemm.bias.contiguous()
        scale = self.scale.contiguous().view(-1)

        y = gemm_scale(x, weight, bias, scale)

        if self.training:
            out = batchnorm_train(
                y, self.bn.weight, self.bn.bias,
                self.bn.running_mean, self.bn.running_var,
                self.eps, self.momentum,
            )
            self.bn.num_batches_tracked.add_(1)
        else:
            out = batchnorm_eval(
                y, self.bn.weight, self.bn.bias,
                self.bn.running_mean, self.bn.running_var,
                self.eps,
            )
        return out